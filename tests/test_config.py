from vllm_tuner.config import BASE_CONFIG, MODEL, validate_config, build_serve_command


def caps(**kw):
    return dict(device_count=1, allowed_tp_sizes=[1], supported_precisions=['bf16'],
                chunked_prefill=True, prefix_caching=True, max_model_len=4096,
                attention_heads=16, kv_heads=8, max_memory_fraction=.65, **kw)


def test_valid_single_device_config():
    assert validate_config(BASE_CONFIG, caps(), {'output_token_limit': 1024, 'input_tokens_max': 200}) == []


def test_rejects_fleet_overallocation_and_unknown_precision():
    errors = validate_config(dict(BASE_CONFIG, tensor_parallel_size=2, precision='w8a8'), caps())
    assert any('fleet' in e for e in errors)
    assert any('evidence' in e for e in errors)


def test_separate_scheduler_and_output_caps():
    assert not validate_config(BASE_CONFIG, caps(), {'output_token_limit': 2048, 'input_tokens_max': 100})
    errors = validate_config(BASE_CONFIG, caps(), {'output_token_limit': 4096, 'input_tokens_max': 100})
    assert any('context length' in e for e in errors)


def test_memory_headroom_nonfinite_and_nonchunked():
    assert validate_config(dict(BASE_CONFIG, gpu_memory_utilization=.9), caps())
    assert validate_config(dict(BASE_CONFIG, gpu_memory_utilization=float('nan')), caps())
    assert validate_config(dict(BASE_CONFIG, chunked_prefill=False), caps())


def test_default_scheduler_is_omitted_and_model_revision_frozen():
    command = build_serve_command(dict(BASE_CONFIG, max_num_seqs=None, max_num_batched_tokens=None), MODEL)
    assert '--max-num-seqs' not in command
    assert '--max-num-batched-tokens' not in command
    assert '--no-enable-prefix-caching' in command
    assert command[command.index('--revision')+1] == MODEL['revision']
    assert command[command.index('--served-model-name')+1] == MODEL['name']


def test_unknown_quantization_and_moving_revision_refused():
    import pytest
    with pytest.raises(ValueError):
        build_serve_command(dict(BASE_CONFIG, precision='w8a8'), MODEL)
    with pytest.raises(ValueError):
        build_serve_command(BASE_CONFIG, dict(MODEL, revision='main'))


def test_malformed_numeric_values_are_validation_errors():
    assert validate_config(dict(BASE_CONFIG, max_num_batched_tokens='512', chunked_prefill=False), caps())
    assert validate_config(dict(BASE_CONFIG, gpu_memory_utilization=True), caps())
    assert validate_config(BASE_CONFIG, caps(), {'output_token_limit': '1024', 'input_tokens_max': 10})


def test_preflight_uses_version_specific_full_help(monkeypatch):
    from types import SimpleNamespace
    from vllm_tuner import config
    flags = ' '.join(x for x in build_serve_command(BASE_CONFIG, MODEL) if x.startswith('--'))
    monkeypatch.setattr(config.shutil, 'which', lambda _: '/fake/vllm')
    for release, expected_help in [('0.10.2', '--help'), ('0.32.0', '--help=all')]:
        calls = []
        def run(argv, **kwargs):
            calls.append(argv)
            if argv[-1] == '--version':
                output = release + ('+cpu' if release == '0.32.0' else '')
            elif argv[1] == 'serve':
                # The old command succeeds but only returns substring matches.
                output = flags if argv[-1] == expected_help else "Parameters matching 'all': --allow-credentials"
            else:
                output = 'benchmark serve help'
            return SimpleNamespace(returncode=0, stdout=output, stderr='')
        monkeypatch.setattr(config.subprocess, 'run', run)
        result = config.preflight_runtime(release)
        assert result['ok'], result['errors']
        assert calls[1] == ['/fake/vllm', 'serve', expected_help]
        assert result['help_commands']['serve'] == calls[1]


def test_preflight_still_fails_closed_on_missing_flags(monkeypatch):
    from types import SimpleNamespace
    from vllm_tuner import config
    monkeypatch.setattr(config.shutil, 'which', lambda _: '/fake/vllm')
    monkeypatch.setattr(config.subprocess, 'run', lambda argv, **kwargs:
        SimpleNamespace(returncode=0, stdout='0.10.2' if argv[-1] == '--version' else '--host', stderr=''))
    result = config.preflight_runtime('0.10.2')
    assert not result['ok']
    assert any('--dtype' in error for error in result['errors'])
