"""ShopRLVE-GYM: Verifiable adaptive-difficulty e-commerce conversation environments for RL training.

Quick start:
    from shop_rlve.server import ShopRLVEEnv
    from shop_rlve.training import run_rollout, DummyModelFn

    env = ShopRLVEEnv(collection="C1")
    obs = env.reset()
    dummy = DummyModelFn(env_id="PD", product_ids=list(env._products_by_id.keys()))
    obs, reward, done, info = env.step(dummy(obs.conversation))
"""

__version__ = "0.1.0"
