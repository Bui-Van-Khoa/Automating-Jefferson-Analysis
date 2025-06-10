from install_requirements import install_requirements

install_requirements()

import os
import subprocess
from transformers import pipeline
from pyannote.audio import Pipeline as DiarizationPipeline
import torchaudio
import opensmile
from tqdm import tqdm
import re


class Jefferson_Transcription:
    def __init__(self, wav_path, num_speakers=2, speaker_names=["A","B"], noise_reduction=False, hf_token=None):
        assert num_speakers == len(speaker_names)
        self.wav_path = wav_path
        self.num_speakers = num_speakers
        self.speaker_names = speaker_names
        self.noise_reduction = noise_reduction
        self.hf_token = hf_token

        # Prepare filtered audio path if needed
        self.filtered_wav_path = None
        if self.noise_reduction:
            print("Performing noise reduction...")
            self.filtered_wav_path = self.apply_noise_reduction()

        # Loads models
        print("Loading diarization model...")
        self.diarization_pipeline = DiarizationPipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=self.hf_token)
        print("Loading ASR model...")
        # Sentence-level timestamps
        #self.asr_pipeline_sentences = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", return_timestamps=True)
        # Word-level timestamps
        self.asr_pipeline_words = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3", return_timestamps="word")        

    # Noise reduction function
    def apply_noise_reduction(self):
        output_dir = "filtered_audio"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(self.wav_path))[0] + "_DeepFilterNet3.wav")

        # Construct and run DeepFilterNet command
        command = ["deepFilter", self.wav_path, "--output-dir", output_dir]
        subprocess.run(command, check=True)

        print(f"Noise-reduced audio saved to: {output_path}")
        return output_path

    # Diarization model function
    def diarize(self):
        wav_for_diarization = self.filtered_wav_path if self.noise_reduction else self.wav_path
        print(f"Performing speaker diarization on: {wav_for_diarization}")
        diarization_result = self.diarization_pipeline(wav_for_diarization, num_speakers=self.num_speakers)
        
        segments = []
        for turn, _, speaker in tqdm(diarization_result.itertracks(yield_label=True), desc="Collecting segments"):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker_id": int(speaker.replace("SPEAKER_", "")),
            })
        return segments

    # transcribe function
    #def transcribe_sentences(self):
        #print("Transcribing audio (this may take a while)...")
        #result = self.asr_pipeline_sentences(self.wav_path)
        #return result["chunks"]

    def transcribe_words(self):
        print("Transcribing audio (this may take a while)...")
        result = self.asr_pipeline_words(self.wav_path)
        return result["chunks"]

    def annotate_loudness(self):
        
        print("Extracting loudness...")

        # Step 1: Per-frame loudness
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        )

        loudness = smile.process_file(self.wav_path)["Loudness_sma3"]

        # Step 2: Overall statistics
        smile_func = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

        func_features = smile_func.process_file(self.wav_path)
        avg_loudness = func_features['loudness_sma3_amean'].iloc[0]
        std_loudness = func_features['loudness_sma3_stddevNorm'].iloc[0]

        # Calculate the threshold: mean + 2*std
        loudness_threshold = avg_loudness + 2 * std_loudness

        # Step 3: Get per-word transcription
        word_chunks = self.transcribe_words() 

        print("Annotating words based on loudness...")

        annotated_chunks = []
        for chunk in word_chunks:
            start, end = chunk["timestamp"]
            condition = (
                (loudness.index.get_level_values(1).total_seconds() >= start) &
                (loudness.index.get_level_values(1).total_seconds() <= end)
            )
            
            segment = loudness[condition]
            word_loudness = segment.mean() if not segment.empty else 0
            emphasized = word_loudness > loudness_threshold

            annotated_chunks.append({
                "text": chunk["text"],
                "timestamp": chunk["timestamp"],
                "emphasized": emphasized
            })

        self.annotated_word_chunks = annotated_chunks  

    def generate_full_annotation(self):
        # 1. Perform speaker diarization
        diarized_segments = self.diarize()
        if not hasattr(self, 'annotated_word_chunks'):
            raise ValueError("Run annotate_loudness() first!")

        
        # 2. Assign each word to the most appropriate speaker segment
        word_assignments = []
        # Loops through each annotated word with its index
        for word_idx, word in enumerate(self.annotated_word_chunks):
            word_start, word_end = word['timestamp']
            # Calculates word's midpont time for late scoring
            word_mid = (word_start + word_end) / 2
            

            best_speaker = None
            best_score = -1
            
            # Find the speaker segment that best matches this word
            for seg in diarized_segments:
                seg_start, seg_end = seg['start'], seg['end']
                
                # Calculate overlap duration
                overlap_start = max(word_start, seg_start)
                overlap_end = min(word_end, seg_end)
                overlap = max(0, overlap_end - overlap_start)
                
                # Bonus if word midpoint falls within segment
                midpoint_bonus = 2 if seg_start <= word_mid <= seg_end else 0
                
                # Additional bonus for early words in segment
                position_bonus = 1 if word_start < seg_start + 0.5 else 0
                
                total_score = overlap + midpoint_bonus + position_bonus
                
                if total_score > best_score:
                    best_score = total_score
                    best_speaker = seg['speaker_id']
            
            # Ensure every word gets assigned to a speaker
            #If no speaker matched, assigns to closest speaker by time distance
            if best_speaker is None and diarized_segments:
                # Fallback: assign to speaker with nearest segment
                time_distances = [
                    (abs(word_start - seg['start']) + abs(word_end - seg['end']), 
                    seg['speaker_id'])
                    for seg in diarized_segments
                ]
                best_speaker = min(time_distances, key=lambda x: x[0])[1]
            
            if best_speaker is not None:
                word_assignments.append({
                    'speaker': self.speaker_names[best_speaker],
                    'text': word['text'],
                    'timestamp': word['timestamp'],
                    'emphasized': word['emphasized'],
                    'original_index': word_idx  # Track original position
                })

        # 3. Sort words by their original order (not just timestamp)
        # This ensures words keep their original sequence from ASR
        word_assignments.sort(key=lambda x: x['original_index'])

        # 4. Process into final transcript with proper formatting
        result = []
        current_speaker = None
        current_line = []
        last_end_time = 0
        min_pause_threshold = 0.08  # 80ms minimum pause to annotate
        speaker_hold_threshold = 0.2  # Time before considering speaker change
        
        for i, word in enumerate(word_assignments):
            word_start, word_end = word['timestamp']
            pause_duration = word_start - last_end_time
            
            # Handle speaker changes
            if current_speaker is None:
                current_speaker = word['speaker']
            elif word['speaker'] != current_speaker:
                # Change speaker only after significant pause or clear turn-taking
                # Checks if the pause before this word exceeds the threshold
                # Detects rapid turn-taking where speakers alternate with short pauses
                if (pause_duration > speaker_hold_threshold or 
                    (i > 0 and word_assignments[i-1]['speaker'] != word['speaker'] and
                    word_start - word_assignments[i-1]['timestamp'][1] > 0.1)):
                    # Finalize current speaker's line
                    if current_line:
                        result.append(self._format_speaker_line(current_speaker, current_line))
                        current_line = []
                    current_speaker = word['speaker']
            
            # Add pause annotation if significant
            if pause_duration >= min_pause_threshold:
                if 0.08 <= pause_duration <= 0.2:
                    current_line.append('(.)')
                elif pause_duration > 0.2:
                    current_line.append(f'({pause_duration:.1f})')
            
            # Add the word (with emphasis if needed)
            text = word['text'].upper() if word['emphasized'] else word['text'].lower()
            current_line.append(text)
            
            last_end_time = word_end
        
        # Add the final line if it exists
        if current_line:
            result.append(self._format_speaker_line(current_speaker, current_line))

        # 5. Post-processing to fix common artifacts
        return result

    def _format_speaker_line(self, speaker, words):
        """Properly format a speaker line with cleaned up spacing"""
        line = ' '.join(words)
        # Clean up space around pause markers
        line = line.replace('( . )', '(.)').replace('( .)', '(.)').replace('(. )', '(.)')
        line = re.sub(r'\(\s*(\d+\.\d+)\s*\)', r'(\1)', line)
        return f"{speaker}: {line}"

  