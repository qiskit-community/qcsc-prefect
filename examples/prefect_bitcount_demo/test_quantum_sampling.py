from examples.prefect_bitcount_demo.quantum_sampling import (
    generate_random_bitstrings,
    resolve_shots,
)


def test_resolve_shots_reads_sampler_option_value():
    shots = resolve_shots(
        sampler_options={"params": {"shots": 321}},
        default_shots=100_000,
    )

    assert shots == 321


def test_generate_random_bitstrings_is_reproducible():
    first = generate_random_bitstrings(bitlen=4, shots=5, seed=24)
    second = generate_random_bitstrings(bitlen=4, shots=5, seed=24)

    assert first == second
    assert len(first) == 5
    assert all(len(bits) == 4 for bits in first)


def test_generate_random_bitstrings_accepts_zero_shots():
    assert generate_random_bitstrings(bitlen=4, shots=0, seed=24) == []
