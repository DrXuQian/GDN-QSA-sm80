"""Explicit child-kernel contracts; do not silently omit a preparation launch."""
FUSED = 'FUSED_SINGLE_KERNEL_V1'
PREPARED = 'PREPARED_AUX_THEN_STATE_V1'
PREPARED_FLAG = '-DGDN_SM90_PRECOMPUTED_AUX=1'


def from_build(build):
    return PREPARED if PREPARED_FLAG in build.get('flags', []) else FUSED


def validate_contracts(receipt):
    contracts = receipt.get('execution_contracts', {})
    if not isinstance(contracts, dict):
        raise ValueError('invalid child-kernel contract manifest')
    for role, contract in contracts.items():
        if role not in ('ours-cuda', 'ours-ppu-source-check', 'ours-candidate') or contract not in (FUSED, PREPARED):
            raise ValueError('unregistered execution contract')
        if contract == PREPARED:
            if role != 'ours-candidate' or not receipt.get('candidate_requires_raw_bit'):
                raise ValueError('two-launch experiment must remain raw-bit candidate-only')
            build = receipt.get('incumbent_builds', {}).get(role, {})
            if from_build(build) != PREPARED:
                raise ValueError('prepared contract lacks actual build-flag binding')
    for role, build in receipt.get('incumbent_builds', {}).items():
        if from_build(build) == PREPARED and contracts.get(role) != PREPARED:
            raise ValueError('two-launch binary cannot inherit single-kernel accounting')
    return contracts
