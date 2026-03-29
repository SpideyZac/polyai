"""Testing code for the simulation worker."""

import time
from multiprocessing import Process, Queue

import simulation_worker
from src_py import viewer_process

# pylint: disable=line-too-long
EXPORT_STRING = "PolyTrack24pdDBvuCABDFAAeL1Ubf5YuEOpZOu9TlLJTI1x80z3XHIY5z83DTiAX0mbmB19KPgF4gN8CODrsPaLf4eUljkI1OO9rFj7CghGZJ5rAKbd2zpCrp4VCzlarogH9tp1fgB"


def main():
    """Main function to run the simulation worker and viewer."""
    # queue: Queue = Queue()
    # viewer = Process(target=viewer_process.run_viewer, args=(queue,), daemon=True)
    # viewer.start()

    # pylint: disable=no-member
    simworker = simulation_worker.SimulationWorkerPy(EXPORT_STRING)
    simworker.init()

    simworker.test_determinism()

    simworker.create_car(0)

    simworker.set_car_controls(
        0,
        # pylint: disable=no-member
        simulation_worker.PlayerControllerPy(True, False, False, False, False),
    )
    start_time = time.time()
    for _ in range(10000):
        simworker.update_car(0)
    end_time = time.time()

    print(
        f"Time taken: {end_time - start_time} or in milliseconds: {(end_time - start_time) * 1000}"
    )

    # time.sleep(10)  # Wait for connections to establish

    # # pylint: disable=line-too-long
    # queue.put(
    #     (
    #         EXPORT_STRING,
    #         "eNod0E0rhFEchvHfGU3GS2OSpY2NsEA2SoRiQch7UWaEJopsmJRvYcFOKUsb2Sr5CrKgyNInsPc8_7O6u8-5r6tOUX5ee30VldrdNZmnVDDIOeMcM8sBY5xEc8gQfckEPwzw0K2WXHDT5jI54zYmDY4i7HKXNEruu4xyFZyzYNblxq2w7LMQlukYzsUwu11jg0V2mIkyyzWW4_0q26xHWE95k5E32QvsSsyzYTUU70krfxXXJdVybqwXzCTDZU_8VnQ06-SxxQungX2r5MznZJL-lNOy8oNPRvjuscRlwVT85D9tmiyr",
    #         26366,
    #     )
    # )

    # time.sleep(10)

    # # pylint: disable=line-too-long
    # queue.put(
    #     (
    #         EXPORT_STRING,
    #         "eNod0E0rhFEchvHfGU3GS2OSpY2NsEA2SoRiQch7UWaEJopsmJRvYcFOKUsb2Sr5CrKgyNInsPc8_7O6u8-5r6tOUX5ee30VldrdNZmnVDDIOeMcM8sBY5xEc8gQfckEPwzw0K2WXHDT5jI54zYmDY4i7HKXNEruu4xyFZyzYNblxq2w7LMQlukYzsUwu11jg0V2mIkyyzWW4_0q26xHWE95k5E32QvsSsyzYTUU70krfxXXJdVybqwXzCTDZU_8VnQ06-SxxQungX2r5MznZJL-lNOy8oNPRvjuscRlwVT85D9tmiyr",
    #         26366,
    #     )
    # )

    # while True:
    #     time.sleep(3600)


if __name__ == "__main__":
    main()
