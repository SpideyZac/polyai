import asyncio
import grpc
import grpc.aio
import argparse
import logging
import inspect
from ray.core.generated import gcs_service_pb2_grpc

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("gcs-proxy")


def build_passthrough_servicer(servicer_class, stub):
    """
    Build a generic passthrough servicer for all services except JobInfoGcsService.
    Logs RPC method names.
    """
    stub_methods = {
        name for name in dir(stub)
        if not name.startswith("_") and callable(getattr(stub, name))
    }
    servicer_methods = {
        name for name, _ in inspect.getmembers(servicer_class, predicate=inspect.isfunction)
        if not name.startswith("_")
    }

    def make_handler(stub_method, method_name):
        async def handler(self, request, context):
            log.info(f"RPC called: {servicer_class.__name__}/{method_name}")
            md = list(context.invocation_metadata())
            return await stub_method(request, metadata=md)
        return handler

    attrs = {}
    for method in stub_methods & servicer_methods:
        stub_method = getattr(stub, method)
        attrs[method] = make_handler(stub_method, method)

    return type(f"{servicer_class.__name__}Passthrough", (servicer_class,), attrs)()


def build_jobinfo_servicer(stub):
    """
    Build a JobInfoGcsServiceServicer that blocks GetNextJobID
    and forwards all other RPCs, logging their names.
    """
    servicer_cls = gcs_service_pb2_grpc.JobInfoGcsServiceServicer
    attrs = {}

    # Intercept GetNextJobID
    async def GetNextJobID(self, request, context):
        log.info("RPC called: JobInfoGcsService/GetNextJobID (blocked)")
        context.set_code(grpc.StatusCode.PERMISSION_DENIED)
    attrs["GetNextJobID"] = GetNextJobID

    # Passthrough all other methods
    for method_name in dir(servicer_cls):
        if method_name.startswith("_") or method_name == "GetNextJobID":
            continue
        base_method = getattr(servicer_cls, method_name, None)
        if not callable(base_method):
            continue

        def make_passthrough(_stub_method_name):
            async def passthrough(self, request, context):
                log.info(f"RPC called: JobInfoGcsService/{_stub_method_name}")
                stub_method = getattr(stub, _stub_method_name)
                md = list(context.invocation_metadata())
                return await stub_method(request, metadata=md)
            return passthrough

        attrs[method_name] = make_passthrough(method_name)

    return type("JobInfoServicerProxy", (servicer_cls,), attrs)()


def discover_services(module):
    """
    Discover all servicers in the gcs_service_pb2_grpc module.
    Returns list of (servicer_class, add_fn, stub_class)
    """
    services = []
    for name, servicer_class in inspect.getmembers(module, inspect.isclass):
        if not name.endswith("Servicer"):
            continue
        base_name = name[: -len("Servicer")]
        add_fn = getattr(module, f"add_{name}_to_server", None)
        stub_class = getattr(module, f"{base_name}Stub", None)
        if add_fn and stub_class:
            services.append((servicer_class, add_fn, stub_class))
    return services


async def serve(listen_port: int, upstream_addr: str):
    upstream_channel = grpc.aio.insecure_channel(upstream_addr)
    server = grpc.aio.server()

    for servicer_class, add_fn, stub_class in discover_services(gcs_service_pb2_grpc):
        stub = stub_class(upstream_channel)
        if servicer_class is gcs_service_pb2_grpc.JobInfoGcsServiceServicer:
            job_servicer = build_jobinfo_servicer(stub)
            add_fn(job_servicer, server)
            log.info("Registered JobInfoServicer with GetNextJobID interception and passthrough")
        else:
            passthrough = build_passthrough_servicer(servicer_class, stub)
            add_fn(passthrough, server)
            log.info(f"Registered passthrough servicer for {servicer_class.__name__}")

    listen_addr = f"[::]:{listen_port}"
    server.add_insecure_port(listen_addr)
    log.info(f"Proxy listening on {listen_addr}, upstream GCS at {upstream_addr}")
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-port", type=int, default=6379)
    parser.add_argument("--upstream", default="localhost:6380")
    args = parser.parse_args()

    asyncio.run(serve(args.listen_port, args.upstream))