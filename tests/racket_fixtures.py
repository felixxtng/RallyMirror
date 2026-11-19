"""Historical sidecars used by current full-review tests; no retired UI required."""
import copy
import shutil
import cv2
import numpy as np
from tests import test_racket_ownership as ownership_fixture
from src.racket.ownership import OwnershipStore, prepare
from src.racket.schema import blank, save_json, read_json, file_hash, digest

def ownership_history(base):
    source,previous=ownership_fixture.OwnershipTests().dataset(base)
    # This clip has one approved location, one approved unknown and one
    # unapproved location; other frames have never been touched.
    images=base/'partial_images';images.mkdir()
    for f in range(6):cv2.imwrite(str(images/f'{f:03}.png'),np.full((100,160,3),100,np.uint8))
    meta=dict(schema_version=1,keypoint_names=['head_center'],annotation_mode='selected_racket',
        source_kind='images',source=str(images),images=[f'{f:03}.png' for f in range(6)],source_hash='partial-fixture',
        source_group='match01',split='train',width=160,height=100,frame_count=6,fps=30,start_frame=0)
    rows={}
    for f in [0,1,2]:
        row=blank(f,['head_center']);row.update(state='visible',approved=f!=2)
        row['keypoints'][0].update(x=65.,y=75.,v=2);rows[str(f)]=row
    rows['1']['state']='unavailable';rows['1']['keypoints'][0].update(x=None,y=None,v=0)
    save_json(source/'clips/partial01/meta.json',meta);save_json(source/'clips/partial01/annotations.json',dict(frames=rows))
    old=OwnershipStore(previous)
    row=old.row('train01_00000');row.update(locked_owner='near_side',other_point=[130.,45.],decision='approved')
    for p in row['players'].values():p['ownership_verified']=True
    old.save('train01_00000',row)
    draft=old.row('train01_00001');draft.update(locked_owner='far_side',other_point=[130.,45.]);old.save('train01_00001',draft)
    unknown=old.row('val01_00000');unknown.update(locked_owner='near_side',decision='unlocatable');old.save('val01_00000',unknown)
    far=copy.deepcopy(row);far.update(frame=1,locked_owner='far_side',locked_reference_sha256=old.tasks['val01_00001']['locked_reference']['row_sha256'])
    old.save('val01_00001',far)
    review=base/'known_exclusions.json'
    save_json(review,[])
    output=base/'all';before={p:file_hash(p) for root in [source,previous] for p in root.rglob('*.json')}
    prepare(source,output,base/'bundle',include_partial=True)
    for clip in old.clips:
        sidecar=previous/'clips'/clip/'ownership.json'
        if sidecar.exists():
            (output/'clips'/clip).mkdir(parents=True,exist_ok=True)
            shutil.copyfile(sidecar,output/'clips'/clip/'ownership.json')
    manifest=read_json(output/'manifest.json')
    manifest.update(workflow='all_existing_centres',carried_training_exclusions=[],
        fixed_original_paired_validation_ids=['val01_00001'],
        locked_imported_reviews={identity:digest(old.row(identity))
            for identity in old.tasks if old.row(identity)['decision']!='pending'})
    save_json(output/'manifest.json',manifest)
    report={}
    return source,previous,output,before,report
