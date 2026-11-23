"""Export immutable organizer images and inspectable provenance without registry credentials."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent/'delivery'/'docker'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    records=[]
    refs=[]
    for n in (7,8,9):
        for role in ('sidecar','environment'):
            ref=f'matbench/task{n:02d}-{role}:20260905'
            info=json.loads(subprocess.check_output(['docker','image','inspect',ref],text=True))[0]
            digest=info['Descriptor']['digest']
            immutable=ref.split(':')[0]+'@'+digest
            subprocess.run(['docker','image','inspect',immutable],check=True,stdout=subprocess.DEVNULL)
            records.append({'task':n,'role':role,'tag':ref,'immutable_reference':immutable,
                            'image_digest':digest,'platform':info['Os']+'/'+info['Architecture'],
                            'image_inspect':info})
            refs.append(ref)
    runner_ref='matbench/harbor-runner:20260906'
    runner_info=json.loads(subprocess.check_output(['docker','image','inspect',runner_ref],text=True))[0]
    runner_digest=runner_info['Descriptor']['digest']
    runner_immutable=runner_ref.split(':')[0]+'@'+runner_digest
    subprocess.run(['docker','image','inspect',runner_immutable],check=True,stdout=subprocess.DEVNULL)
    records.append({'task':None,'role':'harbor_runner','tag':runner_ref,
                    'immutable_reference':runner_immutable,'image_digest':runner_digest,
                    'platform':runner_info['Os']+'/'+runner_info['Architecture'],
                    'image_inspect':runner_info})
    refs += [runner_ref,'mambaorg/micromamba:2.8.1','python:3.11-slim',
             'gogost/gost@sha256:afc0137758ab4ce399d47a299f9abbacbf522b52a17e59cbb4b4e7a1a66e9196']
    archive=OUT/'organizer_images.tar'
    subprocess.run(['docker','image','save','-o',str(archive),*refs],check=True)
    with archive.open('rb') as stream:
        checksum=hashlib.file_digest(stream,'sha256').hexdigest()
    (OUT/'organizer_images.tar.sha256').write_text(checksum+'  organizer_images.tar\n',encoding='utf-8')
    manifest={'distribution':'offline_docker_archive','registry_push_performed':False,
              'archive':archive.name,'archive_sha256':checksum,'archive_bytes':archive.stat().st_size,
              'images':records}
    (OUT/'image_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'archive':str(archive),'sha256':checksum,'bytes':archive.stat().st_size}))

if __name__=='__main__': main()
