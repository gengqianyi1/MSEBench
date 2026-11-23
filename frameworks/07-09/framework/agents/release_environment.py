"""Harbor Docker environment with organizer evidence mounted only in verifiers."""
from pathlib import Path
from harbor.environments.docker.docker import DockerEnvironment
from harbor.models.trial.config import ServiceVolumeConfig


class OrganizerDockerEnvironment(DockerEnvironment):
    def __init__(self, *args, evidence_dir, **kwargs):
        directory = Path(kwargs['environment_dir'])
        if directory.name == 'tests':
            mounts = list(kwargs.get('mounts') or [])
            mounts.append(ServiceVolumeConfig(type='bind', source=Path(evidence_dir).resolve().as_posix(),
                                             target='/organizer_evidence', read_only=True))
            kwargs['mounts'] = mounts
        super().__init__(*args, **kwargs)
