from miditok import REMI, TokenizerConfig
from symusic import Score
config = TokenizerConfig(num_velocities=16, use_chords=True, use_programs=True, use_tempos=True)
tokenizer = REMI(config)

import mido
import os



# pitch to key
pitch_to_key = {
    0: 'C',
    1: 'C#',
    2: 'D',
    3: 'D#',
    4: 'E',
    5: 'F',
    6: 'F#',
    7: 'G',
    8: 'G#',
    9: 'A',
    10: 'A#',
    11: 'B',
}
def get_segmentation(midi_obj):
    pass

def get_key(tokens):
    """
    Enhanced key detection using multiple methods:
    1. Most frequent pitch class
    2. Circle of fifths analysis
    3. Scale degree analysis
    
    Returns a key string that can be used with cdt.Key class
    """
    # Extract all pitch classes
    pitch_classes = []
    for token in tokens:
        if token.startswith('Pitch'):
            pitch = int(token.split('_')[1])
            pitch_classes.append(pitch % 12)
    
    if not pitch_classes:
        return 'C'
    
    # Method 1: Most frequent pitch class
    pitch_count = {}
    for pc in pitch_classes:
        note_name = pitch_to_key[pc]
        pitch_count[note_name] = pitch_count.get(note_name, 0) + 1
    
    most_common_pitch = max(pitch_count, key=pitch_count.get)
    
    # Method 2: Circle of fifths analysis for major/minor determination
    major_keys = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'F']
    minor_keys = ['A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'F', 'C', 'G', 'D']
    
    # Count how many notes fit each key
    key_scores = {}
    
    for key in major_keys:
        key_pc = list(pitch_to_key.keys())[list(pitch_to_key.values()).index(key)]
        # Major scale pattern: 0, 2, 4, 5, 7, 9, 11
        major_scale = [(key_pc + interval) % 12 for interval in [0, 2, 4, 5, 7, 9, 11]]
        score = sum(1 for pc in pitch_classes if pc in major_scale)
        key_scores[f"{key}_major"] = score
    
    for key in minor_keys:
        key_pc = list(pitch_to_key.keys())[list(pitch_to_key.values()).index(key)]
        # Natural minor scale pattern: 0, 2, 3, 5, 7, 8, 10
        minor_scale = [(key_pc + interval) % 12 for interval in [0, 2, 3, 5, 7, 8, 10]]
        score = sum(1 for pc in pitch_classes if pc in minor_scale)
        key_scores[f"{key}_minor"] = score
    
    # Find the best key
    best_key = max(key_scores, key=key_scores.get)
    best_score = key_scores[best_key]
    
    # If the best key has significantly higher score, use it
    if best_score > len(pitch_classes) * 0.6:  # At least 60% of notes fit the key
        return best_key.split('_')[0]  # Return just the note name
    
    # Otherwise, fall back to most common pitch
    return most_common_pitch


def get_key_for_cdt(tokens, key_analysis=None):
    """
    Get key that can be directly used with cdt.Key class.
    Returns the appropriate cdt.Key attribute.
    
    Args:
        tokens: MIDI tokens
        key_analysis: Optional detailed key analysis result to use instead of basic detection
    """
    if key_analysis:
        key_name = key_analysis['key']
    else:
        key_name = get_key(tokens)
    
    # Map key names to cdt.Key attributes
    key_mapping = {
        'C': 'C',
        'C#': 'CSharp',
        'Db': 'DFlat', 
        'D': 'D',
        'D#': 'DSharp',
        'Eb': 'EFlat',
        'E': 'E',
        'F': 'F',
        'F#': 'FSharp',
        'Gb': 'GFlat',
        'G': 'G',
        'G#': 'GSharp',
        'Ab': 'AFlat',
        'A': 'A',
        'A#': 'ASharp',
        'Bb': 'BFlat',
        'B': 'B'
    }
    
    return key_mapping.get(key_name, 'C')


def get_mode_for_cdt(tokens, key_analysis=None):
    """
    Get mode that can be directly used with cdt.Mode class.
    Returns the appropriate cdt.Mode attribute.
    
    Args:
        tokens: MIDI tokens
        key_analysis: Optional detailed key analysis result to use instead of basic detection
    """
    if key_analysis:
        mode_name = key_analysis['mode']
    else:
        # Fallback to basic detection
        key_analysis = get_detailed_key_analysis(tokens)
        mode_name = key_analysis['mode']
    
    # Map mode names to cdt.Mode attributes
    mode_mapping = {
        'major': 'MAJOR',
        'minor': 'MINOR',
        'maj': 'MAJOR',
        'min': 'MINOR'
    }
    
    return mode_mapping.get(mode_name, 'MAJOR')


def get_detailed_key_analysis(tokens):
    """
    Detailed key analysis with confidence scores and mode detection.
    """
    # Extract all pitch classes
    pitch_classes = []
    for token in tokens:
        if token.startswith('Pitch'):
            pitch = int(token.split('_')[1])
            pitch_classes.append(pitch % 12)
    
    if not pitch_classes:
        return {'key': 'C', 'mode': 'major', 'confidence': 0.0, 'analysis': 'No notes found'}
    
    # Count pitch class occurrences
    pc_count = {}
    for pc in pitch_classes:
        pc_count[pc] = pc_count.get(pc, 0) + 1
    
    # Major and minor key analysis
    major_keys = ['C', 'G', 'D', 'A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'F']
    minor_keys = ['A', 'E', 'B', 'F#', 'C#', 'G#', 'D#', 'A#', 'F', 'C', 'G', 'D']
    
    key_analysis = {}
    
    # Analyze major keys
    for key in major_keys:
        key_pc = list(pitch_to_key.keys())[list(pitch_to_key.values()).index(key)]
        major_scale = [(key_pc + interval) % 12 for interval in [0, 2, 4, 5, 7, 9, 11]]
        
        # Count notes that fit the scale
        scale_notes = sum(1 for pc in pitch_classes if pc in major_scale)
        total_notes = len(pitch_classes)
        confidence = scale_notes / total_notes if total_notes > 0 else 0
        
        # Bonus for having the tonic
        tonic_bonus = 0.1 if key_pc in pitch_classes else 0
        
        key_analysis[f"{key}_major"] = {
            'key': key,
            'mode': 'major',
            'confidence': min(confidence + tonic_bonus, 1.0),
            'scale_notes': scale_notes,
            'total_notes': total_notes
        }
    
    # Analyze minor keys
    for key in minor_keys:
        key_pc = list(pitch_to_key.keys())[list(pitch_to_key.values()).index(key)]
        # Natural minor scale pattern: 0, 2, 3, 5, 7, 8, 10
        minor_scale = [(key_pc + interval) % 12 for interval in [0, 2, 3, 5, 7, 8, 10]]
        
        scale_notes = sum(1 for pc in pitch_classes if pc in minor_scale)
        total_notes = len(pitch_classes)
        confidence = scale_notes / total_notes if total_notes > 0 else 0
        
        # Bonus for having the tonic
        tonic_bonus = 0.1 if key_pc in pitch_classes else 0
        
        # Additional bonus for minor third (characteristic of minor keys)
        minor_third = (key_pc + 3) % 12
        minor_third_bonus = 0.05 if minor_third in pitch_classes else 0
        
        # Additional bonus for minor sixth (another characteristic of minor keys)
        minor_sixth = (key_pc + 8) % 12
        minor_sixth_bonus = 0.05 if minor_sixth in pitch_classes else 0
        
        key_analysis[f"{key}_minor"] = {
            'key': key,
            'mode': 'minor',
            'confidence': min(confidence + tonic_bonus + minor_third_bonus + minor_sixth_bonus, 1.0),
            'scale_notes': scale_notes,
            'total_notes': total_notes
        }
    
    # Find the best key
    best_key_name = max(key_analysis, key=lambda k: key_analysis[k]['confidence'])
    best_analysis = key_analysis[best_key_name]
    
    # Handle ties in confidence scores - prefer C major when tied
    best_confidence = best_analysis['confidence']
    tied_keys = [k for k, v in key_analysis.items() if abs(v['confidence'] - best_confidence) < 0.001]
    
    if len(tied_keys) > 1:
        # Priority order: C major > other major keys > minor keys
        if 'C_major' in tied_keys:
            best_key_name = 'C_major'
            best_analysis = key_analysis[best_key_name]
        else:
            # Check if there are both major and minor versions of the same key
            major_keys_in_tie = [k for k in tied_keys if k.endswith('_major')]
            minor_keys_in_tie = [k for k in tied_keys if k.endswith('_minor')]
            
            # If we have both major and minor of the same key, prefer major
            for minor_key in minor_keys_in_tie:
                key_name = minor_key.replace('_minor', '')
                corresponding_major = f"{key_name}_major"
                if corresponding_major in major_keys_in_tie:
                    best_key_name = corresponding_major
                    best_analysis = key_analysis[best_key_name]
                    break
            
            # If no direct major/minor pair, prefer any major key over minor
            if best_key_name.endswith('_minor') and major_keys_in_tie:
                # Find the major key with highest confidence
                best_major = max(major_keys_in_tie, key=lambda k: key_analysis[k]['confidence'])
                best_key_name = best_major
                best_analysis = key_analysis[best_key_name]
    
    # Get top 3 candidates
    sorted_keys = sorted(key_analysis.items(), key=lambda x: x[1]['confidence'], reverse=True)
    top_candidates = sorted_keys[:3]
    
    return {
        'key': best_analysis['key'],
        'mode': best_analysis['mode'],
        'confidence': best_analysis['confidence'],
        'analysis': f"Best fit: {best_analysis['key']} {best_analysis['mode']} (confidence: {best_analysis['confidence']:.2f})",
        'top_candidates': [(name, data['confidence']) for name, data in top_candidates],
        'all_analysis': key_analysis
    }


def analyze_chord_notes(notes, key=None):
    """
    Analyze a list of notes and return possible chord types with their root notes.
    Now considers key context for better analysis.
    
    Args:
        notes: List of MIDI note numbers or note names
        key: Optional key context for better analysis
    
    Returns:
        List of tuples (root_note, chord_type, confidence_score)
    """
    # Convert notes to pitch classes (0-11)
    pitch_classes = set()
    
    for note in notes:
        if isinstance(note, int):
            # MIDI note number
            pitch_classes.add(note % 12)
        elif isinstance(note, str):
            # Note name like 'C', 'C#', 'Db', etc.
            if note in pitch_to_key.values():
                # Find the MIDI number for this note
                for midi_num, note_name in pitch_to_key.items():
                    if note_name == note:
                        pitch_classes.add(midi_num)
                        break
    
    if len(pitch_classes) < 2:
        return []
    
    # Define chord patterns
    chord_patterns = {
        # Major chords
        'major': [0, 4, 7],
        'major7': [0, 4, 7, 11],
        'major9': [0, 4, 7, 11, 2],
        'major6': [0, 4, 7, 9],
        'major6/9': [0, 4, 7, 9, 2],
        
        # Minor chords
        'minor': [0, 3, 7],
        'minor7': [0, 3, 7, 10],
        'minor9': [0, 3, 7, 10, 2],
        'minor6': [0, 3, 7, 9],
        'minor6/9': [0, 3, 7, 9, 2],
        'minor_major7': [0, 3, 7, 11],
        
        # Dominant chords
        'dominant7': [0, 4, 7, 10],
        'dominant9': [0, 4, 7, 10, 2],
        'dominant11': [0, 4, 7, 10, 2, 5],
        'dominant13': [0, 4, 7, 10, 2, 5, 9],
        
        # Diminished chords
        'diminished': [0, 3, 6],
        'diminished7': [0, 3, 6, 9],
        'half_diminished7': [0, 3, 6, 10],
        
        # Augmented chords
        'augmented': [0, 4, 8],
        'augmented7': [0, 4, 8, 10],
        
        # Suspended chords
        'sus2': [0, 2, 7],
        'sus4': [0, 5, 7],
        'sus2/7': [0, 2, 7, 10],
        'sus4/7': [0, 5, 7, 10],
        
        # Power chords
        'power_chord': [0, 7],
        'power_octave': [0, 12],  # Same note, different octave
        'full_octave': [0, 7, 12],  # Root, fifth, octave
        
        # Cluster chords
        'cluster': [0, 1, 2],  # Adjacent notes
        'cluster_wide': [0, 1, 2, 3],  # Wider cluster
        
        # Inversions
        'first_inversion': [0, 3, 8],  # Major first inversion
        'second_inversion': [0, 5, 9],  # Major second inversion
        
        # Root note only
        'root_note': [0],
    }
    
    # Convert pitch classes to list for easier manipulation
    pitch_list = sorted(list(pitch_classes))
    
    # Get key context for better analysis
    key_pc = None
    key_scale = None
    if key and key in pitch_to_key.values():
        key_pc = list(pitch_to_key.keys())[list(pitch_to_key.values()).index(key)]
        # Assume major scale for now (could be enhanced to detect mode)
        key_scale = [(key_pc + interval) % 12 for interval in [0, 2, 4, 5, 7, 9, 11]]
    
    # Find possible chords
    possible_chords = []
    
    for root in range(12):
        # Try each root note
        for chord_name, pattern in chord_patterns.items():
            # Transpose the pattern to the root
            transposed_pattern = [(p + root) % 12 for p in pattern]
            
            # Calculate how many notes match
            matches = len(set(transposed_pattern) & set(pitch_list))
            total_notes = len(pattern)
            
            # Calculate confidence score
            if matches >= 2:  # At least 2 notes must match
                confidence = matches / total_notes
                
                # Bonus for having the root note
                if root in pitch_list:
                    confidence += 0.2
                
                # Bonus for having the third (major/minor indicator)
                if (root + 4) % 12 in pitch_list or (root + 3) % 12 in pitch_list:
                    confidence += 0.1
                
                # Bonus for having the fifth
                if (root + 7) % 12 in pitch_list:
                    confidence += 0.1
                
                # KEY CONTEXT BONUS: Prefer chords that fit the key
                if key_scale and root in key_scale:
                    confidence += 0.15  # Significant bonus for chords in key
                
                # Additional bonus for chord tones that are in the key scale
                if key_scale:
                    chord_tones_in_key = sum(1 for pc in transposed_pattern if pc in key_scale)
                    key_bonus = (chord_tones_in_key / len(transposed_pattern)) * 0.1
                    confidence += key_bonus
                
                # Cap confidence at 1.0
                confidence = min(confidence, 1.0)
                
                root_note = pitch_to_key[root]
                possible_chords.append((root_note, chord_name, confidence))
    
    # Sort by confidence score (highest first)
    possible_chords.sort(key=lambda x: x[2], reverse=True)
    
    # Filter out very low confidence matches
    possible_chords = [chord for chord in possible_chords if chord[2] >= 0.3]
    
    return possible_chords


def get_chord_analysis(tokens, key=None):
    """
    Analyze chords from tokenized MIDI data and return possible chord types.
    
    Args:
        tokens: List of tokens from MIDI tokenizer
        key: Optional key context
    
    Returns:
        Dictionary with chord analysis results
    """
    # Extract notes from tokens
    notes = []
    for token in tokens:
        if token.startswith('Pitch'):
            pitch = int(token.split('_')[1])
            notes.append(pitch)
    
    if not notes:
        return {'error': 'No notes found in tokens'}
    
    # Analyze chords
    chord_analysis = analyze_chord_notes(notes, key)
    
    # Get the most common pitch as key if not provided
    if key is None:
        key = get_key(tokens)
    
    # Group by root note
    chords_by_root = {}
    for root, chord_type, confidence in chord_analysis:
        if root not in chords_by_root:
            chords_by_root[root] = []
        chords_by_root[root].append((chord_type, confidence))
    
    # Sort chords within each root by confidence
    for root in chords_by_root:
        chords_by_root[root].sort(key=lambda x: x[1], reverse=True)
    
    return {
        'key': key,
        'notes': sorted(list(set([note % 12 for note in notes]))),
        'note_names': [pitch_to_key[note % 12] for note in set(notes)],
        'possible_chords': chord_analysis,
        'chords_by_root': chords_by_root,
        'top_chords': chord_analysis[:5]  # Top 5 most likely chords
    }


def get_advanced_chord_analysis(tokens, key=None, style_context=None):
    """
    Advanced chord analysis with style classification and AccoMontage integration.
    
    Args:
        tokens: List of tokens from MIDI tokenizer
        key: Optional key context
        style_context: Optional style context for better analysis
    
    Returns:
        Dictionary with comprehensive chord analysis including style suggestions
    """
    # Get basic chord analysis
    basic_analysis = get_chord_analysis(tokens, key)
    
    if 'error' in basic_analysis:
        return basic_analysis
    
    # Map chord types to AccoMontage chord styles
    chord_style_mapping = {
        'major': 'standard',
        'minor': 'standard',
        'major7': 'seventh',
        'minor7': 'seventh',
        'dominant7': 'seventh',
        'sus2': 'sus2',
        'sus4': 'sus4',
        'power_chord': 'power-chord',
        'power_octave': 'power-octave',
        'full_octave': 'full-octave',
        'cluster': 'cluster',
        'cluster_wide': 'cluster',
        'first_inversion': 'first-inversion',
        'second_inversion': 'second-inversion',
        'root_note': 'root-note',
        'augmented': 'classy',
        'diminished': 'classy',
        'diminished7': 'classy',
        'half_diminished7': 'classy',
        'minor_major7': 'classy',
        'major9': 'emotional',
        'minor9': 'emotional',
        'dominant9': 'emotional',
        'dominant11': 'emotional',
        'dominant13': 'emotional',
    }
    
    # Analyze chord styles
    chord_styles = {}
    for root, chord_type, confidence in basic_analysis['possible_chords']:
        if chord_type in chord_style_mapping:
            style = chord_style_mapping[chord_type]
            if style not in chord_styles:
                chord_styles[style] = []
            chord_styles[style].append((root, chord_type, confidence))
    
    # Sort styles by average confidence
    style_rankings = []
    for style, chords in chord_styles.items():
        avg_confidence = sum(conf for _, _, conf in chords) / len(chords)
        max_confidence = max(conf for _, _, conf in chords)
        style_rankings.append((style, avg_confidence, max_confidence, len(chords)))
    
    style_rankings.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
    
    # Suggest progression styles based on chord analysis
    progression_style_suggestions = []
    
    # Count different chord types
    major_count = sum(1 for _, chord_type, _ in basic_analysis['possible_chords'] 
                     if 'major' in chord_type and 'minor' not in chord_type)
    minor_count = sum(1 for _, chord_type, _ in basic_analysis['possible_chords'] 
                     if 'minor' in chord_type)
    seventh_count = sum(1 for _, chord_type, _ in basic_analysis['possible_chords'] 
                       if '7' in chord_type)
    sus_count = sum(1 for _, chord_type, _ in basic_analysis['possible_chords'] 
                   if 'sus' in chord_type)
    
    # Suggest progression styles
    if seventh_count > 2:
        progression_style_suggestions.append(('r&b', 0.8))
    if sus_count > 1:
        progression_style_suggestions.append(('pop', 0.7))
    if minor_count > major_count:
        progression_style_suggestions.append(('dark', 0.6))
    if major_count > minor_count and seventh_count < 2:
        progression_style_suggestions.append(('pop', 0.7))
    
    # Default to pop if no clear pattern
    if not progression_style_suggestions:
        progression_style_suggestions.append(('pop', 0.5))
    
    progression_style_suggestions.sort(key=lambda x: x[1], reverse=True)
    
    # Add AccoMontage integration suggestions
    accomontage_suggestions = {
        'chord_styles': [style for style, _, _, _ in style_rankings[:3]],
        'progression_styles': [style for style, _ in progression_style_suggestions[:3]],
        'recommended_chord_style': style_rankings[0][0] if style_rankings else 'standard',
        'recommended_progression_style': progression_style_suggestions[0][0] if progression_style_suggestions else 'pop'
    }
    
    return {
        **basic_analysis,
        'chord_styles': chord_styles,
        'style_rankings': style_rankings,
        'progression_style_suggestions': progression_style_suggestions,
        'accomontage_suggestions': accomontage_suggestions
    }


def analyze_midi_structure(tokens):
    """
    Analyze MIDI structure to determine note_shift and segmentation
    
    Args:
        tokens: List of tokens from tokenizer
    
    Returns:
        dict: {
            'total_bars': int,
            'empty_bars': int,
            'content_bars': int,
            'note_shift': int,
            'segmentation': str,
            'first_note_position': int,
            'bar_positions': list
        }
    """
    # Find all bar positions
    bar_positions = [i for i, token in enumerate(tokens) if token.startswith('Bar_None')]
    total_bars = len(bar_positions)
    
    # Find first note position
    first_note_position = None
    for i, token in enumerate(tokens):
        if token.startswith('Pitch_'):
            first_note_position = i
            break
    
    # Calculate empty bars before first note
    empty_bars = 0
    if first_note_position is not None:
        for bar_pos in bar_positions:
            if bar_pos < first_note_position:
                empty_bars += 1
            else:
                break
    
    # Calculate content bars (bars with actual content)
    # add last four bars to avoid missing the last bar
    content_bars = total_bars - empty_bars + 4
    
    # Calculate note_shift (empty_bars * 16 positions per bar)
    note_shift = empty_bars * 16
    
    # Calculate segmentation based on content_bars
    segmentation = calculate_segmentation(content_bars)
    
    return {
        'total_bars': total_bars,
        'empty_bars': empty_bars,
        'content_bars': content_bars,
        'note_shift': note_shift,
        'segmentation': segmentation,
        'first_note_position': first_note_position,
        'bar_positions': bar_positions
    }


def calculate_segmentation(content_bars):
    """
    Calculate segmentation pattern based on content bars
    
    Logic:
    - 1-4 bars: A4
    - 5-8 bars: A8
    - 9+ bars: A8B8A8B8... pattern with remainder handling
    - If remainder is 1-4: use A4
    - If remainder is 5-8: use A8
    
    Args:
        content_bars: Number of bars with actual content
    
    Returns:
        str: Segmentation pattern (e.g., 'A8B8A8B8', 'A4', 'A8', 'A8B4')
    """
    if content_bars <= 0:
        return 'A4'  # Default fallback
    
    if content_bars <= 4:
        # 1-4 bars: A4
        return 'A4'
    
    elif content_bars <= 8:
        # 5-8 bars: A8
        return 'A8'
    
    else:
        # 9+ bars: A8B8A8B8... pattern
        segments = []
        remaining_bars = content_bars
        
        # Add 8-bar segments, alternating between A and B
        segment_count = 0
        while remaining_bars >= 8:
            segment_label = chr(ord('A') + (segment_count % 2))  # Alternate A, B, A, B, ...
            segments.append(f"{segment_label}8")
            remaining_bars -= 8
            segment_count += 1
        
        # Handle remainder
        if remaining_bars > 0:
            if remaining_bars <= 4:
                # 1-4 bars remainder: round up to 4 bars, continue alternating
                adjusted_bars = 4
                segment_label = chr(ord('A') + (segment_count % 2))  # Continue alternating pattern
            else:  # 5-8 bars remainder
                # 5-8 bars remainder: round up to 8 bars, continue alternating
                adjusted_bars = 8
                segment_label = chr(ord('A') + (segment_count % 2))  # Continue alternating pattern
            
            segments.append(f"{segment_label}{adjusted_bars}")
        
        return ''.join(segments)


def get_auto_config(tokens):
    """
    Get automatic configuration for AccoMontage based on MIDI analysis
    
    Args:
        tokens: List of tokens from tokenizer
    
    Returns:
        dict: {
            'note_shift': int,
            'segmentation': str,
            'analysis': dict (from analyze_midi_structure)
        }
    """
    analysis = analyze_midi_structure(tokens)
    
    return {
        'note_shift': analysis['note_shift'],
        'segmentation': analysis['segmentation'],
        'analysis': analysis
    }


def fill_empty_bars_with_chords(midi_file_path, empty_bars, output_path=None):
    """
    Fill empty bars at the beginning with chords from the chord generation using symusic.
    
    Args:
        midi_file_path: Path to the generated MIDI file
        empty_bars: Number of empty bars at the beginning
        output_path: Optional output path, defaults to input path with '_filled' suffix
    
    Returns:
        str: Path to the output file
    """
    # 載入 MIDI 文件
    score = Score(midi_file_path, ttype="tick")
    
    if len(score.tracks) < 2:
        print("Warning: MIDI file doesn't have both piano and chord tracks")
        return midi_file_path
    
    # 找到鋼琴和和弦軌道
    piano_track = None
    chord_track = None
    
    for i, track in enumerate(score.tracks):
        track_name = track.name.lower() if track.name else ''
        if 'piano' in track_name or 'acoustic' in track_name:
            piano_track = i
        elif ('chord' in track_name or 
              track_name == '' or 
              track_name is None or
              i == 1):  # 通常和弦軌道是第二個
            chord_track = i
    
    if piano_track is None or chord_track is None:
        print("Warning: Could not find both piano and chord tracks")
        return midi_file_path
    
    print(f"Found piano track: {piano_track}, chord track: {chord_track}")
    
    # 計算要填充的 bars 數
    bars_to_fill = empty_bars
    # if empty_bars % 4 != 0:
    #     bars_to_fill = (empty_bars // 4) * 4
    #     print(f"Adjusted empty bars from {empty_bars} to {bars_to_fill} to be divisible by 4")
    
    # if bars_to_fill == 0:
    #     print("No empty bars to fill")
    #     return midi_file_path
    
    # 計算時間位置
    ticks_per_beat = score.ticks_per_quarter
    ticks_per_bar = ticks_per_beat * 4  # 4 beats per bar (4/4 time signature)
    empty_ticks = bars_to_fill * ticks_per_bar
    
    print(f"Ticks per quarter: {ticks_per_beat}, Ticks per bar: {ticks_per_bar}")
    print(f"Empty ticks to fill: {empty_ticks}")
    
    # 找到第一個和弦音符的時間
    chord_track_obj = score.tracks[chord_track]
    if not chord_track_obj.notes:
        print("Warning: No chord notes found")
        return midi_file_path
    
    first_chord_time = chord_track_obj.notes[0].start
    print(f"First chord time: {first_chord_time}")
    
    # 收集和弦模式（前 n 個 bars）
    pattern_end_time = first_chord_time + empty_ticks
    pattern_notes = []
    
    for note in chord_track_obj.notes:
        if note.start < pattern_end_time:
            pattern_notes.append(note)
        else:
            break
    
    if not pattern_notes:
        print("Warning: No chord pattern found")
        return midi_file_path
    
    print(f"Pattern notes: {len(pattern_notes)}")
    print(f"Pattern end time: {pattern_end_time}")
    
    # 創建新的 Score
    new_score = Score()
    new_score.ticks_per_quarter = score.ticks_per_quarter
    
    # 複製所有軌道
    for i, track in enumerate(score.tracks):
        if i == chord_track:
            # 對於和弦軌道，創建填充版本
            print(f"Processing chord track {i}")
            
            # 創建模式音符（從時間 0 開始）
            pattern_notes_shifted = []
            for note in pattern_notes:
                new_note = note.copy()
                new_note.start = note.start - first_chord_time
                new_note.end = note.end - first_chord_time
                pattern_notes_shifted.append(new_note)
             
            # 創建原始音符（從 empty_ticks 開始）
            original_notes_shifted = []
            for note in chord_track_obj.notes:
                if note.start >= first_chord_time:
                    new_note = note.copy()
                    new_note.start = note.start - first_chord_time + empty_ticks
                    new_note.end = note.end - first_chord_time + empty_ticks
                    original_notes_shifted.append(new_note)
            
            # 合併模式音符和原始音符
            all_notes = pattern_notes_shifted + original_notes_shifted
            
            # 創建新的軌道
            new_track = track.copy()
            new_track.notes = all_notes
            new_track.notes.sort(key=lambda x: x.start)
            
            new_score.tracks.append(new_track)
        else:
            # 對於其他軌道（如鋼琴），原樣複製
            print(f"Copying track {i} as-is")
            new_score.tracks.append(track.copy())
    
    # 確定輸出路徑
    if output_path is None:
        base_path = os.path.splitext(midi_file_path)[0]
        output_path = f"{base_path}_filled_empty_bars.mid"
    
    # 儲存修改後的 MIDI 文件
    new_score.dump_midi(output_path)
    
    print(f"Filled {bars_to_fill} empty bars with chord pattern")
    print(f"Output saved to: {output_path}")
    
    return output_path

