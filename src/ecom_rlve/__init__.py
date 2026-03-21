"""EcomRLVE-GYM: Verifiable adaptive-difficulty e-commerce conversation environments for RL training.

Quick start:
    from ecom_rlve.server import EcomRLVEEnv
    from ecom_rlve.training import run_rollout, DummyModelFn

    env = EcomRLVEEnv(collection="C1")
    obs = env.reset()
    dummy = DummyModelFn(env_id="PD", product_ids=list(env._products_by_id.keys()))
    obs, reward, done, info = env.step(dummy(obs.conversation))
"""

__version__ = "0.1.0"
