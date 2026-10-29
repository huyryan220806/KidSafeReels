from ml.local_adaptation import speech_groups, split_speech


def fixture_rows():
    return [dict(video_id=f'v{i:03}',source_group_id=f's{i//2}',sha256=f'hash{i}',
                 user_reported_clip_label=(i//2)%2) for i in range(200)]


def test_duplicate_edges_join_entire_source_groups_transitively():
    rows=fixture_rows()
    groups=speech_groups(rows,[dict(left='v000',right='v004'),dict(left='v005',right='v008')])
    assert len({groups[f'v{i:03}'] for i in (0,1,4,5,8,9)})==1


def test_grouped_split_is_deterministic_and_exposed_sources_are_train_only():
    rows=fixture_rows()
    pairs=[dict(left='v000',right='v004')]
    first=split_speech(rows,pairs,['v001'])
    assert first==split_speech(rows,pairs,['v001'])
    assigned={r['video_id']:r['split'] for r in first}
    assert all(assigned[f'v{i:03}']=='train' for i in (0,1,4,5))
    for group in {r['training_group'] for r in first}:
        assert len({r['split'] for r in first if r['training_group']==group})==1
    for split in ('train','valid','test'):
        assert {r['training_label'] for r in first if r['split']==split}=={0,1}
