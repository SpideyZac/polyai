import simulation_worker

simworker = simulation_worker.SimulationWorkerPy()
simworker.init()
print(simworker.test_determinism())
