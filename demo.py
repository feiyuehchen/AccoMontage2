import chorderator as cdt
from miditok import REMI, TokenizerConfig
from symusic import Score
config = TokenizerConfig(num_velocities=16, use_chords=False, use_programs=False)
tokenizer = REMI(config)
from demo_utils import (get_key, 
                       get_chord_analysis, 
                       get_advanced_chord_analysis, 
                       get_detailed_key_analysis, 
                       get_key_for_cdt, 
                       get_mode_for_cdt, 
                       get_auto_config, 
                       fill_empty_bars_with_chords, 
                       export_chords_txt,
                       export_chords_txt_chorder,
                       sync_output_tempo_with_input
                       )

if __name__ == '__main__':
    # Original demo code

    demo_name = '小手'
    input_melody_path = "/home/feiyueh/AccoMontage2/MIDI demos/inputs/midi_with_chord/小手拉大手-梁靜茹-130-C大调.mid"
    cdt.set_melody(input_melody_path)
    # define key and segmentation
    midi_obj = Score(input_melody_path)
    tokens = tokenizer(midi_obj)
    if len(tokens) == 1:
        tokens = tokens[0]
    
    # Optional: Get detailed key analysis
    key_analysis = get_detailed_key_analysis(tokens.tokens)
    print(f"Detailed analysis: {key_analysis['key']} {key_analysis['mode']} (confidence: {key_analysis['confidence']:.2f})")
    
    # Optional: Get cdt.Key and cdt.Mode for direct usage
    cdt_key_attr = get_key_for_cdt(tokens.tokens, key_analysis)
    cdt_mode_attr = get_mode_for_cdt(tokens.tokens, key_analysis)
    print(f"Use: cdt.Key.{cdt_key_attr}, cdt.Mode.{cdt_mode_attr}")

    # Auto-configure note_shift and segmentation
    auto_config = get_auto_config(tokens.tokens)
    print(f"\n=== Auto Configuration ===")
    print(f"Total bars: {auto_config['analysis']['total_bars']}")
    print(f"Empty bars: {auto_config['analysis']['empty_bars']}")
    print(f"Content bars: {auto_config['analysis']['content_bars']}")
    print(f"Note shift: {auto_config['note_shift']} (empty_bars * 16)")
    print(f"Segmentation: {auto_config['segmentation']}")

    # Get actual cdt.Key and cdt.Mode values
    cdt_key_value = getattr(cdt.Key, cdt_key_attr)
    cdt_mode_value = getattr(cdt.Mode, cdt_mode_attr)
    
    cdt.set_meta(tonic=cdt_key_value, mode=cdt_mode_value)
    cdt.set_note_shift(auto_config['note_shift'])
    cdt.set_segmentation(auto_config['segmentation'])
    cdt.set_output_style(cdt.Style.POP_STANDARD)
    # Generate chord progression
    chord_gen = cdt.generate_save(demo_name + '_output_results', task='chord')
    
    # Fill empty bars with chord pattern
    empty_bars = auto_config['analysis']['empty_bars']
    output_file = fill_empty_bars_with_chords(
        demo_name + '_output_results/chord_gen.mid', 
        empty_bars,
        demo_name + '_output_results/chord_gen_filled_empty_bars.mid'
    )
    # Sync output tempo to input tempo
    sync_output_tempo_with_input(input_melody_path, [output_file])
    
    print(f"Final output: {output_file}")
    export_chords_txt_chorder(output_file, demo_name + '_output_results/chord_gen_filled_empty_bars.txt')

