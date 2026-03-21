"""
A gRPC proxy for Ray's GCS server that blocks job, task, and driver worker creation.
Prevents RCE from malicious workers.
"""

import asyncio
import argparse
import logging
import inspect
import grpc
import grpc.aio
from google.protobuf.json_format import MessageToDict
from ray.core.generated import gcs_service_pb2_grpc

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("gcs-proxy")


def to_dict(request):
    """Convert a protobuf message to a dictionary, preserving field names."""
    try:
        return MessageToDict(request, preserving_proto_field_name=True)
    except Exception:  # pylint: disable=broad-exception-caught
        return {}


def is_driver_worker(req):
    """
    Check if the worker being added is a driver worker.
    This is based on the "worker_type" field in the request,
    which should be "DRIVER" for driver workers.
    """
    try:
        wt = req.get("worker_data", {}).get("worker_type")
        return wt == "DRIVER"
    except Exception:  # pylint: disable=broad-exception-caught
        return False


def should_block(servicer, method, req):
    """
    Determine if a request should be blocked based on the servicer, method, and request content.
    """
    if servicer == "TaskInfoGcsServiceServicer" and method == "AddTaskEventData":
        return True

    if servicer == "JobInfoGcsServiceServicer" and method == "AddJob":
        return True

    if servicer == "WorkerInfoGcsServiceServicer" and method == "AddWorkerInfo":
        return is_driver_worker(req)

    if servicer == "JobInfoGcsServiceServicer" and method == "GetNextJobID":
        return True

    if servicer == "InternalKVGcsServiceServicer" and method == "InternalKVPut":
        namespace = req.get("namespace", "")
        if namespace == "ZnVu":
            return True

    return False


def build_servicer(servicer_class, stub):
    """
    Build a proxy servicer that intercepts calls to the specified methods
    and applies blocking logic.
    """
    stub_methods = {
        name
        for name in dir(stub)
        if not name.startswith("_") and callable(getattr(stub, name))
    }
    servicer_methods = {
        name
        for name, _ in inspect.getmembers(servicer_class, predicate=inspect.isfunction)
        if not name.startswith("_")
    }

    def make_handler(stub_method, method_name):
        async def handler(_self, request, context: grpc.aio.ServicerContext):
            req_dict = to_dict(request)

            if should_block(servicer_class.__name__, method_name, req_dict):
                log.warning("[BLOCKED] %s/%s", servicer_class.__name__, method_name)
                log.warning("Called from IP: %s", context.peer())

                context.set_code(grpc.StatusCode.PERMISSION_DENIED)
                context.set_details("Blocked by GCS proxy policy")
                await context.abort(
                    grpc.StatusCode.PERMISSION_DENIED, "Blocked by GCS proxy policy"
                )
                return

            invocation_metadata = context.invocation_metadata()
            if invocation_metadata is not None:
                md = list(invocation_metadata)
            else:
                md = []
            return await stub_method(request, metadata=md)

        return handler

    attrs = {}
    for method in stub_methods & servicer_methods:
        stub_method = getattr(stub, method)
        attrs[method] = make_handler(stub_method, method)

    return type(f"{servicer_class.__name__}Proxy", (servicer_class,), attrs)()


def discover_services(module):
    """
    Discover gRPC services in the specified module by looking for classes
    that end with "Servicer" and their corresponding add_*_to_server functions and Stub classes.
    """
    services = []
    for name, servicer_class in inspect.getmembers(module, inspect.isclass):
        if not name.endswith("Servicer"):
            continue
        base = name[: -len("Servicer")]
        add_fn = getattr(module, f"add_{name}_to_server", None)
        stub = getattr(module, f"{base}Stub", None)
        if add_fn and stub:
            services.append((servicer_class, add_fn, stub))
    return services


async def serve(listen_port: int, upstream: str):
    """
    Start the gRPC proxy server that listens on the specified port
    and forwards requests to the upstream GCS server, applying blocking logic as needed.
    """
    channel = grpc.aio.insecure_channel(upstream)
    server = grpc.aio.server()

    for servicer_class, add_fn, stub_class in discover_services(gcs_service_pb2_grpc):
        stub = stub_class(channel)
        proxy = build_servicer(servicer_class, stub)
        add_fn(proxy, server)
        log.info("Registered %s", servicer_class.__name__)

    addr = f"[::]:{listen_port}"
    server.add_insecure_port(addr)

    log.info("Listening on %s, upstream %s", addr, upstream)

    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-port", type=int, default=6379)
    parser.add_argument("--upstream", default="localhost:6380")
    args = parser.parse_args()

    asyncio.run(serve(args.listen_port, args.upstream))
