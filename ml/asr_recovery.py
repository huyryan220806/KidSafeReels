"""Bounded retry for pathological repeated Vietnamese ASR; no Toxic-score-based selection."""
from collections import Counter

from ml.toxic_baseline import normalize

VERSION = 'asr-repeat-recovery-v1'


def repeated_asr(segments):
    texts = [normalize(s['text']) for s in segments if normalize(s['text'])]
    if len(texts) < 4:
        return False
    # Exact repeated phrases, not common short interjections or accent-folded matches.
    counts = Counter(t for t in texts if len(t.split()) >= 3)
    return any(count >= 4 and count/len(texts) >= .5 for count in counts.values())


def audio_windows(samples, sample_rate=16000, seconds=10, overlap=1):
    if samples <= 0 or sample_rate <= 0 or not 0 <= overlap < seconds:
        raise ValueError('Invalid audio window configuration')
    length, stride = int(seconds*sample_rate), int((seconds-overlap)*sample_rate)
    windows = []
    for start in range(0, samples, stride):
        end = min(samples, start+length)
        windows.append((start, end))
        if end == samples:
            break
    return windows


def transcribe_chunks(asr, audio):
    windows = audio_windows(len(audio)); output = []
    for i, (start, end) in enumerate(windows):
        segments, _ = asr.transcribe(audio[start:end], task='transcribe', beam_size=5, temperature=0,
            condition_on_previous_text=False, language='vi', vad_filter=False)
        for s in segments:
            begin, finish = start/16000+s.start, min(end/16000, start/16000+s.end)
            midpoint = (begin+finish)/2
            left = (start+windows[i-1][1])/32000 if i else 0
            right = (end+windows[i+1][0])/32000 if i+1<len(windows) else len(audio)/16000
            if not s.text.strip() or finish <= begin or not left <= midpoint < right:
                continue
            output.append(dict(start=begin, end=finish, text=s.text, avg_logprob=s.avg_logprob,
                               compression_ratio=s.compression_ratio, no_speech_prob=s.no_speech_prob))
    return output


def recover_repetition(asr, audio, segments, *, language, language_probability):
    """Caller keeps its original language gate. Retry never searches for a higher Toxic score."""
    if language != 'vi' or language_probability < .5 or not repeated_asr(segments):
        return segments, None
    try:
        recovered = transcribe_chunks(asr, audio)
    except (RuntimeError, ValueError):
        return segments, 'Transcript ASR bị lặp; nhận dạng lại chưa thành công, cần nghe đối chiếu.'
    if not recovered or repeated_asr(recovered):
        return segments, 'Transcript ASR bị lặp; chưa khôi phục được lời nói rõ hơn, cần nghe đối chiếu.'
    return recovered, 'ASR phát hiện transcript lặp và đã nhận dạng lại theo đoạn ngắn; cần đối chiếu audio khi kết quả chưa rõ.'
