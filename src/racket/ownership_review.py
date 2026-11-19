"""Explicit correction queues for saved training pairs, with resumable review progress."""
from pathlib import Path

from .schema import digest,file_hash,read_json,save_json

FORMAT='racket_ownership_correction_review_v001'


class CorrectionReview:
    def __init__(self,store,path):
        self.store=store;self.path=Path(path).resolve();self.data=read_json(self.path)
        if self.data.get('format')!=FORMAT:raise ValueError('Unsupported correction review queue')
        if Path(self.data['dataset_root']).resolve()!=store.root:raise ValueError('Review belongs to a different dataset')
        if self.data['dataset_manifest_sha256']!=file_hash(store.root/'manifest.json'):
            raise ValueError('Correction review source manifest changed')
        entries=self.data['entries'];self.entries={e['id']:e for e in entries}
        if not entries or len(entries)!=len(self.entries):raise ValueError('Empty or duplicated correction frame IDs')
        for identity in self.entries:
            task=store.tasks.get(identity)
            if not task or task['split']!='train':raise ValueError('Correction queue must contain registered training frames only')
            if identity in store.manifest.get('locked_imported_reviews',{}):
                raise ValueError('Cannot reopen a locked imported review: '+identity)
            if digest(task)!=self.entries[identity]['task_sha256']:raise ValueError('Correction task changed: '+identity)
        self.progress_path=self.path.with_name(self.path.stem+'_progress.json')
        self.progress=dict(queue_sha256=file_hash(self.path),reviewed={})
        if self.progress_path.exists():
            self.progress=read_json(self.progress_path)
            if self.progress.get('queue_sha256')!=file_hash(self.path):raise ValueError('Correction review queue changed after progress was saved')
            if not set(self.progress['reviewed'])<=set(self.entries):raise ValueError('Unknown frame in correction progress')

    def remaining(self,queue):
        return [k for k in queue if self.store.row(k)['decision']=='pending' or
                self.progress['reviewed'].get(k)!=digest(self.store.row(k))]

    def confirm(self,identity):
        if identity not in self.entries:raise ValueError('Frame is outside the correction queue')
        if self.store.row(identity)['decision']=='pending':raise ValueError('Complete or exclude the frame before confirming review')
        self.progress['reviewed'][identity]=digest(self.store.row(identity))
        save_json(self.progress_path,self.progress)


def prepare_review(store,report,output):
    output=Path(output)
    if output.exists():raise FileExistsError('Review already exists; resume without replacing it')
    entries=[]
    for flag in report['outside_body_derived_context']:
        identity=flag['id'];task=store.tasks[identity]
        entries.append(dict(id=identity,side=flag['side'],reason=flag['visual_review'],
                            task_sha256=digest(task),original_row=store.row(identity)))
    data=dict(format=FORMAT,dataset_root=str(store.root),dataset_manifest_sha256=file_hash(store.root/'manifest.json'),
              entries=entries,policy='Only listed training frames; source points and imported reviews stay locked. Saved rows are not reset. Confirm with Space.')
    save_json(output,data)
    return CorrectionReview(store,output)
