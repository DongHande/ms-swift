# Copyright (c) ModelScope Contributors. All rights reserved.
import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'scripts/utils/build_cached_dataset/distributed_cached_dataset.py'
spec = importlib.util.spec_from_file_location('role_cached_builder', SCRIPT)
builder = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = builder
spec.loader.exec_module(builder)


@pytest.fixture
def config(monkeypatch):
    for name in ('DATA_SEED', 'DATASET_SHUFFLE', 'SPLIT_DATASET_RATIO', 'MS_SWIFT_REPO', 'ROLE_LOSS_CONFIG',
                 'LOSS_SCALE', 'RUN_ID'):
        monkeypatch.delenv(name, raising=False)
    return builder.BuildConfig.from_file(SCRIPT.with_name(builder.DEFAULT_CONFIG_NAME))


def test_example_resolves_current_repository(config):
    assert config.ms_swift_repo.resolve() == ROOT
    assert config.role_loss_config.is_file()
    assert config.loss_scale == 'role'
    assert config.materialize_role_weights


@pytest.mark.parametrize('changes', [
    {'data_seed': 42},
    {'dataset_shuffle': False},
    {'split_dataset_ratio': 0.1},
    {'max_length': 1024},
    {'agent_template': 'hermes'},
])
def test_fingerprint_changes_with_training_semantics(config, changes):
    assert config.build_fingerprint != dataclasses.replace(config, **changes).build_fingerprint


def test_fingerprint_tracks_role_config_contents(config, tmp_path):
    path = tmp_path / 'role.json'
    path.write_text(config.role_loss_config.read_text())
    first = dataclasses.replace(config, role_loss_config=path).build_fingerprint
    data = json.loads(path.read_text())
    data['assistant_weights']['answer'] = 0.25
    path.write_text(json.dumps(data))
    second = dataclasses.replace(config, role_loss_config=path).build_fingerprint
    assert first != second


def test_cached_shard_rejects_missing_or_mismatched_manifest(config, tmp_path):
    assert not config.cached_shard_is_ready(tmp_path)
    (tmp_path / '_SUCCESS').touch()
    with pytest.raises(SystemExit, match='缺少构建指纹'):
        config.cached_shard_is_ready(tmp_path)
    manifest = tmp_path / 'build_manifest.json'
    manifest.write_text(json.dumps({'build_fingerprint': 'old'}))
    with pytest.raises(SystemExit, match='指纹不一致'):
        config.cached_shard_is_ready(tmp_path)
    manifest.write_text(json.dumps({'build_fingerprint': config.build_fingerprint}))
    assert config.cached_shard_is_ready(tmp_path)


@pytest.mark.parametrize('host', ['-oProxyCommand=bad', 'host/name', 'host;command', 'host with spaces'])
def test_worker_hosts_reject_option_and_path_injection(config, tmp_path, host):
    path = tmp_path / 'hosts.txt'
    path.write_text(host)
    launcher = builder.RemoteWorkerLauncher(dataclasses.replace(config, ip_file=path))
    with pytest.raises(SystemExit, match='invalid worker host'):
        launcher.read_hosts()


def test_worker_hosts_accept_names_and_addresses(config, tmp_path):
    path = tmp_path / 'hosts.txt'
    path.write_text('# comment\nworker-1.example.org\n192.0.2.1\n2001:db8::1\n')
    launcher = builder.RemoteWorkerLauncher(dataclasses.replace(config, ip_file=path))
    assert launcher.read_hosts() == ['worker-1.example.org', '192.0.2.1', '2001:db8::1']


def test_runtime_preserves_cudnn_path(config, monkeypatch):
    monkeypatch.setenv('LD_LIBRARY_PATH', '/existing/lib')
    updated = dataclasses.replace(config, cuda_fallback_ld_path='/configured/cudnn/lib')
    values = updated.runtime_env_for_worker()
    assert values['LD_LIBRARY_PATH'] == '/configured/cudnn/lib:/existing/lib'


def test_fingerprint_covers_agent_and_template_implementations(config):
    hashes = config.build_metadata['ms_swift_source_hashes']
    assert 'swift/agent_template/base.py' in hashes
    assert 'swift/agent_template/hermes.py' in hashes
    assert 'swift/template/register.py' in hashes
    assert 'swift/dataset/preprocessor/core.py' in hashes
    assert 'requirements/framework.txt' in hashes
