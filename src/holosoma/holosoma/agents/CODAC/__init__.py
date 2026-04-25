from holosoma.agents.CODAC.CODAC_agent import CODACAgent, CODACEnv

# Backward compatibility aliases for earlier names.
OFFLINESACAgent = CODACAgent
OFFLINESACEnv = CODACEnv
OfflineSACAgent = CODACAgent
OfflineSACEnv = CODACEnv

__all__ = [
    "CODACAgent",
    "CODACEnv",
    "OfflineSACAgent",
    "OfflineSACEnv",
    "OFFLINESACAgent",
    "OFFLINESACEnv",
]
