"""T15: runner, exact resume và prune; chỉ fixture packed, không mạng."""
from __future__ import annotations
import copy
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch
import torch
from transformers import Wav2Vec2Config, Wav2Vec2Model
from scripts import train_xlsr as runner
from scripts.train_b0 import _set_seed
from tests.test_b0_v2_runner import _write_artifact
from src.data.packed import PackedManifestDataset
from src.training.full_training import GenerationStore, run_training
from src.training.sampler import ResumableBatchSampler

ROOT = Path(__file__).resolve().parents[1]

def equal(test: unittest.TestCase, left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        test.assertTrue(torch.equal(left, right))
    elif isinstance(left, dict):
        test.assertEqual(set(left), set(right))
        for key in left:
            equal(test, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        test.assertEqual(len(left), len(right))
        for a, b in zip(left, right, strict=True):
            equal(test, a, b)
    else:
        test.assertEqual(left, right)

class RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.plan = json.loads((ROOT/'configs/xlsr_training.json').read_text())
        self.artifact = self.root/'packed'
        _write_artifact(self.artifact, split='closed_train', rows=4, manifest_sha256='a'*64, offset=3)
        self.dev_artifact=self.root/'dev-packed'
        _write_artifact(self.dev_artifact,split='closed_dev',rows=4,manifest_sha256='b'*64,offset=6)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def fixture(self, name: str, *, resume: bool = False, limit: int = 99,
                can_start: Callable[[str, int], bool] | None = None,
                identity_change: dict[str, Any] | None = None, prune: bool = True) -> dict[str, Any]:
        _set_seed(2026)
        config = json.loads((ROOT/'configs/xlsr_smoke.json').read_text())['backbone']
        config.update(num_hidden_layers=3, layerdrop=.1, apply_spec_augment=True)
        model = runner.build_model(Wav2Vec2Model(Wav2Vec2Config(**config)), {'layers':2}, 2026)
        runner.effective_modes(model)
        optimizer = torch.optim.Adam(model.optimizer_groups(encoder_lr=1e-5, classifier_lr=.001))
        sampler = ResumableBatchSampler(dataset_size=4, batch_size=2, seed=2026)
        ds = PackedManifestDataset(self.artifact, training=True, seed=2026, amplitude_policy='peak')
        dev = PackedManifestDataset(self.dev_artifact, training=False, seed=2026, amplitude_policy='peak')
        store = GenerationStore(self.root/name, prune=prune)
        identity = {'seed':2026,'lr':1e-5,'layers':2,'artifact':'a'*64,'weights':'b'*64}
        identity.update(identity_change or {})
        def step(indices: list[int], epoch: int) -> float:
            ds.set_epoch(epoch)
            batch = runner.default_collate([ds[i] for i in indices])
            model.train(); optimizer.zero_grad(set_to_none=True)
            waveform = runner.apply_xlsr_waveform_normalization(batch['waveform'], runner.NORMALIZATION)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(waveform),batch['label'])
            loss.backward(); optimizer.step()
            return float(loss.detach())
        def evaluate() -> dict[str, Any]:
            # Đánh giá thật trên packed fixture hai nhãn; mode eval tác động RNG/lịch sử.
            return runner.evaluate(model, dev, torch.device('cpu'), 2, [0,1,2,3])
        try:
            result = run_training(model=model,optimizer=optimizer,sampler=sampler,identity=identity,
                store=store,train_step=step,evaluate=evaluate,max_epochs=2,patience=2,
                development_samples=4,session_steps=limit,checkpoint_interval=1,resume=resume,
                trainable_only=True,can_start=can_start,retain_evaluations=True)
            payload = store.load()
            self.assertEqual(set(payload['model']), {n for n,p in model.named_parameters() if p.requires_grad})
            return {'result':result,'payload':payload}
        finally:
            ds.close()
            dev.close()

    def compare(self, left: dict[str, Any], right: dict[str, Any]) -> None:
        a, b = copy.deepcopy(left['payload']), copy.deepcopy(right['payload'])
        a['controller']['best']['generation'] = None
        b['controller']['best']['generation'] = None
        # Chỉ loại số đo wall-clock: toàn metric/loss/state còn lại vẫn đối chiếu từng bit.
        for payload in [a,b]:
            payload['controller']['best']['development'].pop('timing',None)
            for row in payload['controller']['history']:
                row.get('development',{}).pop('timing',None)
        equal(self, a, b)  # model, optimizer, sampler, mọi RNG, history, best và threshold

    def test_exact_resume_mid_epoch(self) -> None:
        full=self.fixture('full'); self.fixture('split',limit=1)
        self.compare(full,self.fixture('split',resume=True))

    def test_exact_resume_before_evaluation(self) -> None:
        full=self.fixture('full')
        paused=self.fixture('split',can_start=lambda phase,count:phase!='evaluation')
        self.assertTrue(paused['payload']['controller']['needs_evaluation'])
        self.compare(full,self.fixture('split',resume=True))

    def test_exact_resume_before_first_training(self) -> None:
        full=self.fixture('full')
        paused=self.fixture('split',can_start=lambda phase,count:False)
        self.assertEqual(paused['payload']['controller']['global_step'],0)
        self.compare(full,self.fixture('split',resume=True))

    def test_exact_resume_after_evaluation_before_next_training(self) -> None:
        full=self.fixture('full'); calls=0
        def budget(phase: str,count: int) -> bool:
            nonlocal calls
            if phase=='training': calls+=1
            return calls<=2
        paused=self.fixture('split',can_start=budget)
        self.assertEqual(paused['payload']['controller']['completed_epochs'],1)
        self.compare(full,self.fixture('split',resume=True))

    def test_exact_resume_multiple_pauses_and_pending_prune(self) -> None:
        full=self.fixture('full'); self.fixture('split',limit=1)
        self.fixture('split',resume=True,can_start=lambda phase,count:phase!='evaluation')
        self.fixture('split',resume=True,limit=1)
        self.compare(full,self.fixture('split',resume=True))

    def test_prune_matches_unpruned_training(self) -> None:
        self.compare(self.fixture('full',prune=False),self.fixture('pruned'))

    def test_trainable_checkpoint_rejects_missing_name(self) -> None:
        self.fixture('split',limit=1); store=GenerationStore(self.root/'split')
        payload=store.load(); payload['model'].pop(next(iter(payload['model'])));store.commit(payload)
        with self.assertRaisesRegex(ValueError,'Danh sách'):
            self.fixture('split',resume=True)

    def test_trainable_checkpoint_rejects_shape(self) -> None:
        self.fixture('split',limit=1); store=GenerationStore(self.root/'split')
        payload=store.load(); key=next(iter(payload['model']));payload['model'][key]=torch.zeros(1);store.commit(payload)
        with self.assertRaisesRegex(ValueError,'Tensor'):
            self.fixture('split',resume=True)

    def test_trainable_checkpoint_rejects_dtype(self) -> None:
        self.fixture('split',limit=1); store=GenerationStore(self.root/'split')
        payload=store.load(); key=next(iter(payload['model']));payload['model'][key]=payload['model'][key].double();store.commit(payload)
        with self.assertRaisesRegex(ValueError,'Tensor'):
            self.fixture('split',resume=True)

    def test_modes_d029_keep_raw_config(self) -> None:
        self.fixture('split',limit=1)
        self.assertFalse(self.plan['pretrained']['effective_apply_spec_augment'])
        self.assertTrue(self.plan['pretrained']['configured_apply_spec_augment'])

    def test_pretrained_tamper_rejected(self) -> None:
        weights=self.root/'weights';weights.mkdir()
        for name in ['pytorch_model.bin','config.json','preprocessor_config.json']:
            (weights/name).write_bytes(b'bad')
        with self.assertRaisesRegex(ValueError,'Checksum pretrained'):
            runner.verify_pretrained(weights,self.plan)

    def test_budget_reserves_evaluation_and_commit_time(self) -> None:
        budget=runner.SessionBudget(1,evaluation_seconds=3000,step_seconds=1,clock=lambda:0)
        self.assertTrue(budget.can_start('evaluation',8))
        budget.deadline=3100
        self.assertFalse(budget.can_start('evaluation',8))
        self.assertTrue(budget.can_start('training',1))

    def test_packed_test_rejected_before_opening_waveforms(self) -> None:
        with patch.object(runner,'PackedManifestDataset') as loader:
            with self.assertRaises(ValueError):
                runner.open_artifact(self.artifact,self.plan,training=False,seed=2026)
            loader.assert_not_called()

    def test_smoke_balanced_indices(self) -> None:
        ds=PackedManifestDataset(self.artifact,training=True,amplitude_policy='peak')
        try:
            self.assertEqual(runner.smoke_indices(ds,4),[0,1,2,3])
            ds.split='closed_test'
            with self.assertRaisesRegex(ValueError,'test'):runner.smoke_indices(ds,4)
        finally:ds.close()

    def test_d023_transfer_roundtrip_trainable_only(self) -> None:
        import importlib.util
        spec=importlib.util.spec_from_file_location('transfer',ROOT/'deploy/kaggle/resume_checkpoint.py')
        transfer=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(transfer)
        before=self.fixture('split',limit=1)
        store=GenerationStore(self.root/'split');payload=store.load()
        payload['identity']['source_tree_sha256']='f'*64
        store.commit(payload)
        target=self.root/'package'
        manifest=transfer.build_payload(store.root,target,'fixture/checkpoint')
        restored=self.root/'restored'
        transfer.restore_payload(target,restored,transfer.digest(target/transfer.MANIFEST_NAME),
            1,'f'*64,self.root/'restored-report.json')
        equal(self,GenerationStore(restored).load(),store.load())
        (target/manifest['files'][0]['file']).write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError,'checksum'):
            transfer.restore_payload(target,self.root/'bad',transfer.digest(target/transfer.MANIFEST_NAME),
                1,'f'*64,self.root/'bad-report.json')

    def test_preflight_reports_actual_training_gate(self) -> None:
        import importlib.util
        spec=importlib.util.spec_from_file_location('xlsr_preflight',ROOT/'.kaggle/xlsr-d028/preflight.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        for ready in [False,True]:
            with self.subTest(ready=ready):
                plan=copy.deepcopy(self.plan);plan['implementation']['ready_for_training']=ready
                fields=module.readiness_report_fields(plan)
                self.assertIs(fields['ready_for_training'],ready)
                self.assertEqual(fields['status'],'passed_local_preflight_gate_open' if ready else 'passed_local_preflight_gate_closed')

    def test_kaggle_profiles_are_private_offline_and_use_existing_data(self) -> None:
        config=json.loads((ROOT/'.kaggle/xlsr-d028/project.json').read_text())
        self.assertFalse(config['enable_internet'])
        self.assertEqual(config['data_dataset_ref'],'ngctnnguyn1111/doanttnt-t14-packed')
        metadata=json.loads((ROOT/'.kaggle/xlsr-d028/workflow/kernel/kernel-metadata.json').read_text())
        self.assertTrue(metadata['is_private'])
        self.assertIn('ngctnnguyn1111/doanttnt-t14-weights',metadata['dataset_sources'])
        self.assertIn('ngctnnguyn1111/doanttnt-t14-offline-deps',metadata['dataset_sources'])

    def test_kaggle_bootstrap_embeds_exact_d023_transfer(self) -> None:
        import base64
        import re
        config=json.loads((ROOT/'.kaggle/xlsr-d028/project.json').read_text())
        command=config['profiles'][config['default_profile']][0]
        encoded=re.search(r"b64decode\('([^']+)'\)",command[2]).group(1)
        bootstrap=base64.b64decode(encoded).decode()
        compile(bootstrap,'bootstrap','exec')
        transfer=(ROOT/'deploy/kaggle/resume_checkpoint.py').read_bytes()
        self.assertIn(base64.b64encode(transfer).decode(),bootstrap)
        self.assertIn('--no-index',bootstrap)
        self.assertIn('scripts/train_xlsr.py',bootstrap)

    def test_large_session_time_budget_commits_k_steps_and_resumes_exactly(self) -> None:
        full=self.fixture('full',limit=9930)
        now=[1000.0]
        budget=runner.SessionBudget(11,evaluation_seconds=20,step_seconds=10,
            deadline_unix=1170,clock=lambda:now[0])
        def can_start(phase: str,count: int) -> bool:
            allowed=budget.can_start(phase,count)
            if allowed:now[0]+=20 if phase=='evaluation' else 10
            return allowed
        paused=self.fixture('split',limit=9930,can_start=can_start)
        self.assertEqual(paused['payload']['controller']['global_step'],3)
        self.assertEqual(paused['payload']['controller']['completed_epochs'],1)
        self.compare(full,self.fixture('split',resume=True,limit=9930))

    def test_large_session_evaluates_and_trains_next_epoch_without_restart(self) -> None:
        result=self.fixture('full',limit=9930)
        state=result['payload']['controller']
        self.assertEqual(result['result']['steps_this_invocation'],4)
        self.assertEqual([row['global_step'] for row in state['history']],[2,4])
        self.assertEqual(state['completed_epochs'],2)

    def test_different_session_steps_resume_identity_unchanged(self) -> None:
        full=self.fixture('full',limit=9930)
        first=self.fixture('split',limit=1)
        resumed=self.fixture('split',resume=True,limit=9930)
        self.assertEqual(first['payload']['identity'],resumed['payload']['identity'])
        self.compare(full,resumed)

    def test_cli_forwards_session_steps_without_identity_change(self) -> None:
        plan=copy.deepcopy(self.plan)
        # Chỉ fixture giả có gate mở, không đổi plan thật và không đọc VSASV/pretrained.
        plan['implementation']['ready_for_training']=True
        for prefix,path in [('train',self.artifact),('development',self.dev_artifact)]:
            metadata=json.loads((path/'metadata.json').read_text())
            plan['data'][prefix+'_samples']=4
            plan['data'][prefix+'_metadata_sha256']=runner.sha256(path/'metadata.json')
            plan['data'][prefix+'_manifest_sha256']=metadata['source_manifest_sha256']
            plan['data'][prefix+'_artifact_waveforms_sha256']=metadata['waveforms_sha256']
        plan_path=self.root/'fixture-plan.json';plan_path.write_text(json.dumps(plan),encoding='utf-8')
        def backbone(*args: Any,**kwargs: Any) -> Wav2Vec2Model:
            _set_seed(2026)
            config=json.loads((ROOT/'configs/xlsr_smoke.json').read_text())['backbone']
            config.update(num_hidden_layers=3,layerdrop=.1,apply_spec_augment=True)
            return Wav2Vec2Model(Wav2Vec2Config(**config))
        common=['--plan',str(plan_path),'--run-id','X1-L2-lr1e-5','--training-seed','2026',
            '--device','cpu','--pretrained-dir',str(self.root/'no-real-weights'),
            '--train-artifact',str(self.artifact),'--dev-artifact',str(self.dev_artifact)]
        observed=[]
        real_run=runner.run_training
        def capture(**kwargs: Any) -> dict[str,Any]:
            observed.append((kwargs['session_steps'],copy.deepcopy(kwargs['identity'])))
            return real_run(**kwargs)
        with patch.object(runner,'validate_plan',return_value={'run_id':'X1-L2-lr1e-5','layers':2,'encoder_lr':1e-5}), \
             patch.object(runner,'verify_pretrained',return_value={'fixture':'random'}), \
             patch.object(runner,'load_xlsr_backbone',side_effect=backbone), \
             patch.object(runner,'verify_packed_artifact'), \
             patch.object(runner,'_sample_selection_sha256',return_value=plan['data']['sample_selection_sha256']), \
             patch.object(runner,'run_training',side_effect=capture):
            self.assertEqual(runner.main(common+['--run-root',str(self.root/'full'),'--report-json',str(self.root/'full.json')]),0)
            self.assertEqual(runner.main(common+['--session-steps','1','--run-root',str(self.root/'split'),'--report-json',str(self.root/'split.json')]),0)
            self.assertEqual(runner.main(common+['--session-steps','9930','--resume','--run-root',str(self.root/'split'),'--report-json',str(self.root/'split.json')]),0)
        self.assertEqual([x[0] for x in observed],[512,1,9930])
        self.assertEqual(observed[1][1],observed[2][1])
        self.assertNotIn('"timing"',json.dumps(observed[2][1]))
        self.compare({'payload':GenerationStore(self.root/'full').load()},
                     {'payload':GenerationStore(self.root/'split').load()})

    def test_evaluate_timing_with_fake_clock_and_history(self) -> None:
        import itertools
        _set_seed(2026)
        config=json.loads((ROOT/'configs/xlsr_smoke.json').read_text())['backbone']
        model=runner.XLSRBinaryClassifier(Wav2Vec2Model(Wav2Vec2Config(**config)),num_unfrozen_layers=1)
        ds=PackedManifestDataset(self.dev_artifact,training=False,amplitude_policy='peak')
        ticks=itertools.count()
        try:
            result=runner.evaluate(model,ds,torch.device('cpu'),2,clock=lambda:float(next(ticks)))
        finally:ds.close()
        timing=result['timing']
        self.assertEqual(set(timing),{'loader_seconds','model_seconds','total_seconds','samples','throughput_samples_per_second'})
        self.assertEqual(timing['loader_seconds'],2.)
        self.assertEqual(timing['model_seconds'],2.)
        self.assertEqual(timing['total_seconds'],9.)
        self.assertEqual(timing['samples'],4)
        self.assertAlmostEqual(timing['throughput_samples_per_second'],4/9)
        payload=self.fixture('history')['payload']
        self.assertIn('timing',payload['controller']['history'][0]['development'])
        self.assertNotIn('timing',payload['identity'])

    def test_progress_store_logs_committed_steps_and_keeps_payload(self) -> None:
        import io,contextlib
        observed=runner.InvocationTiming(clock=lambda:1.)
        for _ in range(64):observed.record_step(.25,.75)
        store=runner.ProgressGenerationStore(self.root/'progress',timing=observed,prune=False)
        state={'global_step':64,'completed_epochs':0,'best':None,'losses':[.5]*64}
        payload={'controller':state,'sampler':{'epoch':0}}
        stream=io.StringIO()
        with contextlib.redirect_stdout(stream):
            committed=store.commit(payload)
        record=json.loads(stream.getvalue())
        self.assertEqual(record['event'],'checkpoint')
        self.assertEqual(record['global_step'],64)
        self.assertEqual(record['epoch'],1)
        self.assertEqual(record['mean_loss_last_64'],.5)
        self.assertEqual(record['mean_step_seconds'],1.)
        self.assertEqual(committed['controller'],state)
        self.assertEqual(payload['controller'],state)

    def test_cli_report_contains_invocation_timing_and_progress(self) -> None:
        # Dùng đường CLI thật với backbone/packed fixture đã có trong test trước.
        import io,contextlib
        stream=io.StringIO()
        # Test CLI hiện có gọi run_training qua capture riêng; theo dõi report file bằng write_text.
        original_write=Path.write_text
        emitted=[]
        def write(path: Path,data: str,*args: Any,**kwargs: Any) -> int:
            if path.name in ['full.json','split.json']:
                parsed=json.loads(data)
                emitted.append(parsed)
            return original_write(path,data,*args,**kwargs)
        with patch.object(Path,'write_text',write),contextlib.redirect_stdout(stream):
            self.test_cli_forwards_session_steps_without_identity_change()
        self.assertEqual(len(emitted),3)
        for report in emitted:
            timing=report['timing']
            for key in ['rehash_train_seconds','rehash_dev_seconds','backbone_load_seconds','model_build_seconds',
                        'train_loader_seconds','train_model_seconds','train_seconds','mean_step_seconds',
                        'wall_seconds','seconds_left_at_exit']:
                self.assertTrue(torch.isfinite(torch.tensor(timing[key])))
                self.assertGreaterEqual(timing[key],0)
            self.assertEqual(timing['train_steps_this_invocation'],report['steps_this_invocation'])
            self.assertNotIn('timing',report['identity'])
            self.assertEqual(len(timing['evaluations_this_invocation']),
                             report['controller']['completed_epochs']-(1 if report is emitted[-1] else 0))
        records=[json.loads(line) for line in stream.getvalue().splitlines() if line.startswith('{')]
        events=[x.get('event') for x in records]
        for event in ['rehash_train_start','rehash_train_end','rehash_dev_start','rehash_dev_end',
                      'backbone_load_start','model_build_end','train_start','checkpoint',
                      'evaluate_start','evaluate_end','stop']:
            self.assertIn(event,events)
        self.assertIn('paused',[x.get('reason') for x in records if x.get('event')=='stop'])
        self.assertTrue(all('timestamp_utc' in x for x in records if 'event' in x))


class SessionControlTests(unittest.TestCase):
    def bootstrap(self) -> Any:
        import importlib.util
        spec=importlib.util.spec_from_file_location('xlsr_bootstrap',ROOT/'.kaggle/xlsr-d028/bootstrap.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module

    def test_absolute_deadline_overrides_hours_and_uses_unix_clock(self) -> None:
        budget=runner.SessionBudget(.001,evaluation_seconds=20,step_seconds=10,
            deadline_unix=5000,clock=lambda:1000)
        self.assertEqual(budget.deadline,5000)
        self.assertTrue(budget.can_start('evaluation',8))

    def test_expired_absolute_deadline_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError,'hạn|Hạn'):
            runner.SessionBudget(11,evaluation_seconds=20,step_seconds=10,
                deadline_unix=999,clock=lambda:1000)

    def test_nonfinite_absolute_deadline_rejected(self) -> None:
        for value in [float('nan'),float('inf'),-float('inf'),0.0,-1.0]:
            with self.subTest(value=value),self.assertRaises(ValueError):
                runner.SessionBudget(11,evaluation_seconds=20,step_seconds=10,
                    deadline_unix=value,clock=lambda:1000)

    def test_deadline_commit_margin_is_120_seconds(self) -> None:
        now=[1000.0]
        budget=runner.SessionBudget(11,evaluation_seconds=20,step_seconds=10,
            deadline_unix=1140,clock=lambda:now[0])
        self.assertTrue(budget.can_start('evaluation',8))
        now[0]+=.01
        self.assertFalse(budget.can_start('evaluation',8))

    def test_default_relative_budget_keeps_monotonic_clock(self) -> None:
        budget=runner.SessionBudget(1,evaluation_seconds=20,step_seconds=10,clock=lambda:1000)
        self.assertEqual(budget.deadline,4600)

    def test_cli_invalid_deadline_rejected_before_pretrained_read(self) -> None:
        for value in ['1','nan','inf','-inf']:
            with self.subTest(value=value),patch.object(runner,'verify_pretrained') as verify:
                with self.assertRaisesRegex(ValueError,'Hạn Unix'):
                    runner.main(['--run-id','X0','--training-seed','2026','--device','cpu',
                        '--pretrained-dir','fixture','--dry-run','--deadline-unix='+value])
                verify.assert_not_called()

    def test_absolute_deadline_default_clock_is_wall_clock(self) -> None:
        with patch.object(runner.time,'time',return_value=1000), \
             patch.object(runner.time,'monotonic',return_value=999999):
            budget=runner.SessionBudget(11,evaluation_seconds=20,step_seconds=10,deadline_unix=2000)
            self.assertTrue(budget.can_start('evaluation',8))

    def test_cli_session_steps_defaults_to_512(self) -> None:
        args=runner.build_parser().parse_args(['--run-id','X0','--training-seed','2026','--device','cpu','--pretrained-dir','fixture'])
        self.assertEqual(args.session_steps,512)
        self.assertIsNone(args.deadline_unix)

    def test_cli_session_steps_rejects_nonpositive(self) -> None:
        parser=runner.build_parser()
        for value in ['0','-1','1.2']:
            with self.subTest(value=value),self.assertRaises(SystemExit):
                parser.parse_args(['--run-id','X0','--training-seed','2026','--device','cpu','--pretrained-dir','fixture','--session-steps',value])

    def test_bootstrap_command_uses_beginning_of_session(self) -> None:
        import argparse
        module=self.bootstrap()
        plan=json.loads((ROOT/'configs/xlsr_training.json').read_text())
        args=argparse.Namespace(run_id='X1-L2-lr1e-5',training_seed='2026',
            data_root=Path('/data'),dry_run=False,resume_manifest_sha256=None)
        with patch.object(module.time,'time',return_value=2200):
            command=module.build_runner_command(args,plan=plan,python='python',weights=Path('/weights'),
                run_root=Path('/run'),report=Path('/report.json'),started_unix=1000)
        self.assertEqual(command[command.index('--session-steps')+1],'9930')
        self.assertEqual(float(command[command.index('--deadline-unix')+1]),40600)

    def test_bootstrap_main_captures_clock_before_setup(self) -> None:
        import argparse
        import io,contextlib
        from types import SimpleNamespace
        module=self.bootstrap();now=[1000.0];commands=[];stdout=io.StringIO()
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);manifest=base/'offline_manifest.json'
            manifest.write_text('{"files":{}}',encoding='utf-8')
            args=argparse.Namespace(run_id='X0',training_seed='2026',data_root=base/'data',
                working_root=base/'working',dry_run=False,resume_manifest_sha256=None,expected_step=None)
            def checked(command: list[str],env: dict[str,str],
                        on_exit: Callable[[int],None] | None = None) -> None:
                commands.append(command)
                if 'pip' in command:now[0]+=1200
                if on_exit is not None:on_exit(0)
            def rglob(path: Path,pattern: str) -> list[Path]:
                return [manifest] if pattern=='offline_manifest.json' else [base/'pytorch_model.bin']
            with contextlib.redirect_stdout(stdout),patch.object(module,'time',SimpleNamespace(time=lambda:now[0]),create=True), \
                 patch.object(module.argparse.ArgumentParser,'parse_args',return_value=args), \
                 patch.object(module.Path,'rglob',rglob), \
                 patch.object(module.venv,'EnvBuilder'), \
                 patch.object(module.subprocess,'check_output',return_value=b'unchanged'), \
                 patch.object(module,'checked',side_effect=checked):
                self.assertEqual(module.main(),0)
        milestones=[json.loads(line) for line in stdout.getvalue().splitlines()]
        self.assertEqual([x['event'] for x in milestones],
                         ['bootstrap_start','offline_checksum_end','venv_pip_end','runner_start','runner_end'])
        self.assertEqual(milestones[-1]['exit_code'],0)
        self.assertEqual(milestones[-1]['elapsed_seconds'],1200)
        command=commands[-1]
        self.assertIn('--session-steps',command)
        self.assertEqual(command[command.index('--session-steps')+1],'9930')
        self.assertEqual(float(command[command.index('--deadline-unix')+1]),40600)

    def test_bootstrap_dry_run_does_not_add_training_controls(self) -> None:
        import argparse
        module=self.bootstrap()
        plan=json.loads((ROOT/'configs/xlsr_training.json').read_text())
        args=argparse.Namespace(run_id='X0',training_seed='2026',data_root=Path('/data'),
            dry_run=True,resume_manifest_sha256=None)
        command=module.build_runner_command(args,plan=plan,python='python',weights=Path('/weights'),
            run_root=Path('/run'),report=Path('/report.json'),started_unix=1000)
        self.assertIn('--dry-run',command)
        self.assertNotIn('--session-steps',command)
        self.assertNotIn('--deadline-unix',command)

class TimingContractTests(unittest.TestCase):
    def test_invocation_report_fake_clock(self) -> None:
        import itertools
        ticks=itertools.count()
        timing=runner.InvocationTiming(clock=lambda:float(next(ticks)))
        timing.phase('rehash_train',lambda:None)
        timing.phase('rehash_dev',lambda:None)
        timing.phase('backbone_load',lambda:None)
        timing.phase('model_build',lambda:None)
        timing.record_step(2.,3.);timing.record_step(4.,5.)
        timing.evaluations.append({'total_seconds':7.,'samples':4})
        plan=json.loads((ROOT/'configs/xlsr_training.json').read_text())
        budget=runner.SessionBudget(1,evaluation_seconds=20,step_seconds=10,clock=lambda:1000)
        report=timing.report(wall_seconds=12,deadline_unix=None,budget=budget,plan=plan,layers=0)
        self.assertEqual(report['rehash_train_seconds'],1.)
        self.assertEqual(report['backbone_load_seconds'],1.)
        self.assertEqual(report['train_steps_this_invocation'],2)
        self.assertEqual(report['train_seconds'],14.)
        self.assertEqual(report['mean_step_seconds'],7.)
        self.assertEqual(report['seconds_left_at_exit'],3600)
        self.assertEqual(len(report['evaluations_this_invocation']),1)
        self.assertEqual(report['estimates']['dev_full_seconds'],3015.)

    def test_cuda_synchronize_only_for_cuda(self) -> None:
        with patch.object(runner.torch.cuda,'synchronize') as sync:
            runner.synchronize(torch.device('cpu'));sync.assert_not_called()
            runner.synchronize(torch.device('cuda:0'));sync.assert_called_once_with(torch.device('cuda:0'))

    def test_progress_is_flushed_and_has_utc_timestamp(self) -> None:
        from datetime import datetime
        with patch('builtins.print') as output:
            runner.progress('checkpoint',global_step=64)
        self.assertTrue(output.call_args.kwargs['flush'])
        event=json.loads(output.call_args.args[0])
        self.assertEqual(event['event'],'checkpoint')
        self.assertEqual(datetime.fromisoformat(event['timestamp_utc']).utcoffset().total_seconds(),0)

    def test_bootstrap_pure_milestones_and_runner_exit_code(self) -> None:
        import importlib.util
        spec=importlib.util.spec_from_file_location('timed_bootstrap',ROOT/'.kaggle/xlsr-d028/bootstrap.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        names=['bootstrap_start','offline_checksum_end','venv_pip_end','restore_end','runner_start','runner_end']
        events=[module.bootstrap_event(name,1000,1000+i*10) for i,name in enumerate(names)]
        self.assertEqual([x['event'] for x in events],names)
        self.assertEqual([x['elapsed_seconds'] for x in events],[0,10,20,30,40,50])
        captured=[]
        with patch.object(module.subprocess,'run',return_value=type('Result',(),{'returncode':7})()):
            with self.assertRaisesRegex(RuntimeError,'7'):
                module.checked(['python','fixture'],{},on_exit=captured.append)
        self.assertEqual(captured,[7])


class PruneTests(unittest.TestCase):
    def test_preserves_latest_best_pending(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store=GenerationStore(Path(temp),prune=True)
            payload={'checkpoint_version':4,'controller':{'best':None},'model':{}}
            store.commit(payload)
            pending=json.loads((store.root/'LATEST.json').read_text())['latest']['generation']
            payload['controller']['pending_generation']=pending
            store.commit(payload)
            payload['controller']['best']={'generation':None}
            result=store.commit(payload,new_best=True)
            best=result['controller']['best']['generation']; payload=result
            store.commit(payload)
            latest=json.loads((store.root/'LATEST.json').read_text())['latest']['generation']
            self.assertEqual({p.name for p in store.root.glob('g-*')},{latest,best,pending})

    def test_default_prune_false_keeps_every_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store=GenerationStore(Path(temp))
            for _ in range(4):store.commit({'controller':{'best':None}})
            self.assertEqual(len(list(store.root.glob('g-*'))),4)

    def test_invalid_pending_path_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError,'generation'):
                GenerationStore(Path(temp),prune=True).commit({'controller':{'best':None,'pending_generation':'../bad'}})

# Mỗi biến thể là một test độc lập để số lượng và lỗi có tên rõ ràng.
def plan_case(section: str, key: str, value: Any) -> Callable[[RunnerTests], None]:
    def test(self: RunnerTests) -> None:
        plan=copy.deepcopy(self.plan);plan[section][key]=value
        with self.assertRaises(ValueError):runner.validate_plan(plan,'X1-L2-lr1e-5',2026)
    return test
for section,key,value in [('data','data_seed',2027),('data','use_test',True),('data','backend','parquet'),
    ('data','development_split','closed_test'),('training','batch_size',8),('training','classifier_learning_rate',.01),
    ('training','encoder_learning_rates',[.0001]),('training','max_epochs',3),('evaluation','patience',1),
    ('evaluation','strict_improvement',False),('model','classifier_dropout',.3),('model','amp',True),
    ('pretrained','effective_layerdrop',.1),('pretrained','effective_apply_spec_augment',True),
    ('pretrained','configured_apply_spec_augment',False)]:
    setattr(RunnerTests,'test_plan_rejects_'+section+'_'+key,plan_case(section,key,value))

def identity_case(key: str, value: Any) -> Callable[[RunnerTests], None]:
    def test(self: RunnerTests) -> None:
        self.fixture('split',limit=1)
        with self.assertRaisesRegex(ValueError,'Identity'):
            self.fixture('split',resume=True,identity_change={key:value})
    return test
for key,value in [('seed',2027),('lr',3e-5),('layers',1),('artifact','c'*64),('weights','d'*64)]:
    setattr(RunnerTests,'test_identity_rejects_'+key,identity_case(key,value))
