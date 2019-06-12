# encoding: utf-8
"""
This module contains chord evaluation functionality.

It provides the evaluation measures used for the MIREX ACE task, and
tries to follow [1]_ and [2]_ as closely as possible.

Notes
-----
This implementation tries to follow the references and their implementation
(e.g., https://github.com/jpauwels/MusOOEvaluator for [2]_). However, there
are some known (and possibly some unknown) differences. If you find one not
listed in the following, please file an issue:

 - Detected chord segments are adjusted to fit the length of the annotations.
   In particular, this means that, if necessary, filler segments of 'no chord'
   are added at beginnings and ends. This can result in different segmentation
   scores compared to the original implementation.

References
----------
.. [1] Christopher Harte, "Towards Automatic Extraction of Harmony Information
       from Music Signals." Dissertation,
       Department for Electronic Engineering, Queen Mary University of London,
       2010.
.. [2] Johan Pauwels and Geoffroy Peeters.
       "Evaluating Automatically Estimated Chord Sequences."
       In Proceedings of ICASSP 2013, Vancouver, Canada, 2013.

"""

import numpy as np
import pandas as pd
import mir_eval


CHORD_DTYPE = [('root', np.int),
               ('bass', np.int),
               ('intervals', np.int, (12,)),
               ('is_major',np.bool)]

CHORD_ANN_DTYPE = [('start', np.float),
                   ('end', np.float),
                   ('chord', CHORD_DTYPE)]

NO_CHORD = (-1, -1, np.zeros(12, dtype=np.int), False)
UNKNOWN_CHORD = (-1, -1, np.ones(12, dtype=np.int) * -1, False)

PITCH_CLASS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def idx_to_chord(idx):
    if idx == 24:
        return "-"
    elif idx == 25:
        return u"\u03B5"

    minmaj = idx % 2
    root = idx // 2

    return PITCH_CLASS[root] + ("M" if minmaj == 0 else "m")

class Chords:

    def __init__(self):
        self._shorthands = {
            'maj': self.interval_list('(1,3,5)'),
            'min': self.interval_list('(1,b3,5)'),
            'dim': self.interval_list('(1,b3,b5)'),
            'aug': self.interval_list('(1,3,#5)'),
            'maj7': self.interval_list('(1,3,5,7)'),
            'min7': self.interval_list('(1,b3,5,b7)'),
            '7': self.interval_list('(1,3,5,b7)'),
            '6': self.interval_list('(1,6)'),  # custom
            '5': self.interval_list('(1,5)'),
            '4': self.interval_list('(1,4)'),  # custom
            '1': self.interval_list('(1)'),
            'dim7': self.interval_list('(1,b3,b5,bb7)'),
            'hdim7': self.interval_list('(1,b3,b5,b7)'),
            'minmaj7': self.interval_list('(1,b3,5,7)'),
            'maj6': self.interval_list('(1,3,5,6)'),
            'min6': self.interval_list('(1,b3,5,6)'),
            '9': self.interval_list('(1,3,5,b7,9)'),
            'maj9': self.interval_list('(1,3,5,7,9)'),
            'min9': self.interval_list('(1,b3,5,b7,9)'),
            'sus2': self.interval_list('(1,2,5)'),
            'sus4': self.interval_list('(1,4,5)'),
            '11': self.interval_list('(1,3,5,b7,9,11)'),
            'min11': self.interval_list('(1,b3,5,b7,9,11)'),
            '13': self.interval_list('(1,3,5,b7,13)'),
            'maj13': self.interval_list('(1,3,5,7,13)'),
            'min13': self.interval_list('(1,b3,5,b7,13)')
        }

    def chords(self, labels):

        """
        Transform a list of chord labels into an array of internal numeric
        representations.

        Parameters
        ----------
        labels : list
            List of chord labels (str).

        Returns
        -------
        chords : numpy.array
            Structured array with columns 'root', 'bass', and 'intervals',
            containing a numeric representation of chords.

        """
        crds = np.zeros(len(labels), dtype=CHORD_DTYPE)
        cache = {}
        for i, lbl in enumerate(labels):
            cv = cache.get(lbl, None)
            if cv is None:
                cv = self.chord(lbl)
                cache[lbl] = cv
            crds[i] = cv

        return crds

    def label_error_modify(self, label):
        if label == 'Emin/4': label = 'E:min/4'
        elif label == 'A7/3': label = 'A:7/3'
        elif label == 'Bb7/3': label = 'Bb:7/3'
        elif label == 'Bb7/5': label = 'Bb:7/5'
        elif label.find(':') == -1:
            if label.find('min') != -1:
                label = label[:label.find('min')] + ':' + label[label.find('min'):]
        return label

    def chord(self, label):
        """
        Transform a chord label into the internal numeric represenation of
        (root, bass, intervals array).

        Parameters
        ----------
        label : str
            Chord label.

        Returns
        -------
        chord : tuple
            Numeric representation of the chord: (root, bass, intervals array).

        """

        try:
            is_major = False

            if label == 'N':
                return NO_CHORD
            if label == 'X':
                return UNKNOWN_CHORD

            label = self.label_error_modify(label)

            c_idx = label.find(':')
            s_idx = label.find('/')

            if c_idx == -1:
                quality_str = 'maj'
                if s_idx == -1:
                    root_str = label
                    bass_str = ''
                else:
                    root_str = label[:s_idx]
                    bass_str = label[s_idx + 1:]
            else:
                root_str = label[:c_idx]
                if s_idx == -1:
                    quality_str = label[c_idx + 1:]
                    bass_str = ''
                else:
                    quality_str = label[c_idx + 1:s_idx]
                    bass_str = label[s_idx + 1:]

            root = self.pitch(root_str)
            bass = self.interval(bass_str) if bass_str else 0
            ivs = self.chord_intervals(quality_str)
            ivs[bass] = 1

            if 'min' in quality_str:
                is_major = False
            else:
                is_major = True

        except Exception as e:
            print(e, label)

        return root, bass, ivs, is_major

    _l = [0, 1, 1, 0, 1, 1, 1]
    _chroma_id = (np.arange(len(_l) * 2) + 1) + np.array(_l + _l).cumsum() - 1

    def modify(self, base_pitch, modifier):
        """
        Modify a pitch class in integer representation by a given modifier string.

        A modifier string can be any sequence of 'b' (one semitone down)
        and '#' (one semitone up).

        Parameters
        ----------
        base_pitch : int
            Pitch class as integer.
        modifier : str
            String of modifiers ('b' or '#').

        Returns
        -------
        modified_pitch : int
            Modified root note.

        """
        for m in modifier:
            if m == 'b':
                base_pitch -= 1
            elif m == '#':
                base_pitch += 1
            else:
                raise ValueError('Unknown modifier: {}'.format(m))
        return base_pitch

    def pitch(self, pitch_str):
        """
        Convert a string representation of a pitch class (consisting of root
        note and modifiers) to an integer representation.

        Parameters
        ----------
        pitch_str : str
            String representation of a pitch class.

        Returns
        -------
        pitch : int
            Integer representation of a pitch class.

        """
        return self.modify(self._chroma_id[(ord(pitch_str[0]) - ord('C')) % 7],
                      pitch_str[1:]) % 12

    def interval(self, interval_str):
        """
        Convert a string representation of a musical interval into a pitch class
        (e.g. a minor seventh 'b7' into 10, because it is 10 semitones above its
        base note).

        Parameters
        ----------
        interval_str : str
            Musical interval.

        Returns
        -------
        pitch_class : int
            Number of semitones to base note of interval.

        """
        for i, c in enumerate(interval_str):
            if c.isdigit():
                return self.modify(self._chroma_id[int(interval_str[i:]) - 1],
                              interval_str[:i]) % 12

    def interval_list(self, intervals_str, given_pitch_classes=None):
        """
        Convert a list of intervals given as string to a binary pitch class
        representation. For example, 'b3, 5' would become
        [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0].

        Parameters
        ----------
        intervals_str : str
            List of intervals as comma-separated string (e.g. 'b3, 5').
        given_pitch_classes : None or numpy array
            If None, start with empty pitch class array, if numpy array of length
            12, this array will be modified.

        Returns
        -------
        pitch_classes : numpy array
            Binary pitch class representation of intervals.

        """
        if given_pitch_classes is None:
            given_pitch_classes = np.zeros(12, dtype=np.int)
        for int_def in intervals_str[1:-1].split(','):
            int_def = int_def.strip()
            if int_def[0] == '*':
                given_pitch_classes[self.interval(int_def[1:])] = 0
            else:
                given_pitch_classes[self.interval(int_def)] = 1
        return given_pitch_classes

    # mapping of shorthand interval notations to the actual interval representation

    def chord_intervals(self, quality_str):
        """
        Convert a chord quality string to a pitch class representation. For
        example, 'maj' becomes [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0].

        Parameters
        ----------
        quality_str : str
            String defining the chord quality.

        Returns
        -------
        pitch_classes : numpy array
            Binary pitch class representation of chord quality.

        """
        list_idx = quality_str.find('(')
        if list_idx == -1:
            return self._shorthands[quality_str].copy()
        if list_idx != 0:
            ivs = self._shorthands[quality_str[:list_idx]].copy()
        else:
            ivs = np.zeros(12, dtype=np.int)


        return self.interval_list(quality_str[list_idx:], ivs)

    def load_chords(self, filename):
        """
        Load chords from a text file.

        The chord must follow the syntax defined in [1]_.

        Parameters
        ----------
        filename : str
            File containing chord segments.

        Returns
        -------
        crds : numpy structured array
            Structured array with columns "start", "end", and "chord",
            containing the beginning, end, and chord definition of chord
            segments.

        References
        ----------
        .. [1] Christopher Harte, "Towards Automatic Extraction of Harmony
               Information from Music Signals." Dissertation,
               Department for Electronic Engineering, Queen Mary University of
               London, 2010.

        """
        start, end, chord_labels = [], [], []
        with open(filename, 'r') as f:
            for line in f:
                if line:

                    splits = line.split()
                    if len(splits) == 3:

                        s = splits[0]
                        e = splits[1]
                        l = splits[2]

                        start.append(float(s))
                        end.append(float(e))
                        chord_labels.append(l)

        crds = np.zeros(len(start), dtype=CHORD_ANN_DTYPE)
        crds['start'] = start
        crds['end'] = end
        crds['chord'] = self.chords(chord_labels)

        return crds

    def reduce_to_triads(self, chords, keep_bass=False):
        """
        Reduce chords to triads.

        The function follows the reduction rules implemented in [1]_. If a chord
        chord does not contain a third, major second or fourth, it is reduced to
        a power chord. If it does not contain neither a third nor a fifth, it is
        reduced to a single note "chord".

        Parameters
        ----------
        chords : numpy structured array
            Chords to be reduced.
        keep_bass : bool
            Indicates whether to keep the bass note or set it to 0.

        Returns
        -------
        reduced_chords : numpy structured array
            Chords reduced to triads.

        References
        ----------
        .. [1] Johan Pauwels and Geoffroy Peeters.
               "Evaluating Automatically Estimated Chord Sequences."
               In Proceedings of ICASSP 2013, Vancouver, Canada, 2013.

        """
        unison = chords['intervals'][:, 0].astype(bool)
        maj_sec = chords['intervals'][:, 2].astype(bool)
        min_third = chords['intervals'][:, 3].astype(bool)
        maj_third = chords['intervals'][:, 4].astype(bool)
        perf_fourth = chords['intervals'][:, 5].astype(bool)
        dim_fifth = chords['intervals'][:, 6].astype(bool)
        perf_fifth = chords['intervals'][:, 7].astype(bool)
        aug_fifth = chords['intervals'][:, 8].astype(bool)
        no_chord = (chords['intervals'] == NO_CHORD[-1]).all(axis=1)

        reduced_chords = chords.copy()
        ivs = reduced_chords['intervals']

        ivs[~no_chord] = self.interval_list('(1)')
        ivs[unison & perf_fifth] = self.interval_list('(1,5)')
        ivs[~perf_fourth & maj_sec] = self._shorthands['sus2']
        ivs[perf_fourth & ~maj_sec] = self._shorthands['sus4']

        ivs[min_third] = self._shorthands['min']
        ivs[min_third & aug_fifth & ~perf_fifth] = self.interval_list('(1,b3,#5)')
        ivs[min_third & dim_fifth & ~perf_fifth] = self._shorthands['dim']

        ivs[maj_third] = self._shorthands['maj']
        ivs[maj_third & dim_fifth & ~perf_fifth] = self.interval_list('(1,3,b5)')
        ivs[maj_third & aug_fifth & ~perf_fifth] = self._shorthands['aug']

        if not keep_bass:
            reduced_chords['bass'] = 0
        else:
            # remove bass notes if they are not part of the intervals anymore
            reduced_chords['bass'] *= ivs[range(len(reduced_chords)),
                                          reduced_chords['bass']]
        # keep -1 in bass for no chords
        reduced_chords['bass'][no_chord] = -1

        return reduced_chords

    def convert_to_id(self, root, is_major):
        if root == -1:
            return 24
        else:
            if is_major:
                return root * 2
            else:
                return root * 2 + 1

    def get_converted_chord(self, filename):
        loaded_chord = self.load_chords(filename)
        triads = self.reduce_to_triads(loaded_chord['chord'])

        df = self.assign_chord_id(triads)
        df['start'] = loaded_chord['start']
        df['end'] = loaded_chord['end']

        return df

    def assign_chord_id(self, entry):
        # maj, min chord only
        # if you want to add other chord, change this part and get_converted_chord(reduce_to_triads)
        df = pd.DataFrame(data=entry[['root', 'is_major']])
        df['chord_id'] = df.apply(lambda row: self.convert_to_id(row['root'], row['is_major']), axis=1)
        return df

    def convert_to_id_voca(self, root, quality):
        if root == -1:
            return 169
        else:
            if quality == 'min':
                return root * 14
            elif quality == 'maj':
                return root * 14 + 1
            elif quality == 'dim':
                return root * 14 + 2
            elif quality == 'aug':
                return root * 14 + 3
            elif quality == 'min6':
                return root * 14 + 4
            elif quality == 'maj6':
                return root * 14 + 5
            elif quality == 'min7':
                return root * 14 + 6
            elif quality == 'minmaj7':
                return root * 14 + 7
            elif quality == 'maj7':
                return root * 14 + 8
            elif quality == '7':
                return root * 14 + 9
            elif quality == 'dim7':
                return root * 14 + 10
            elif quality == 'hdim7':
                return root * 14 + 11
            elif quality == 'sus2':
                return root * 14 + 12
            elif quality == 'sus4':
                return root * 14 + 13
            else:
                return 168

    def get_converted_chord_voca(self, filename):
        loaded_chord = self.load_chords(filename)
        triads = self.reduce_to_triads(loaded_chord['chord'])
        df = pd.DataFrame(data=triads[['root', 'is_major']])

        (ref_intervals, ref_labels) = mir_eval.io.load_labeled_intervals(filename)
        ref_labels = self.lab_file_error_modify(ref_labels)
        idxs = list()
        for i in ref_labels:
            chord_root, quality, scale_degrees, bass = mir_eval.chord.split(i, reduce_extended_chords=True)
            root, bass, ivs, is_major = self.chord(i)
            idxs.append(self.convert_to_id_voca(root=root, quality=quality))
        df['chord_id'] = idxs

        df['start'] = loaded_chord['start']
        df['end'] = loaded_chord['end']

        return df

    def lab_file_error_modify(self, ref_labels):
        for i in range(len(ref_labels)):
            if ref_labels[i][-2:] == ':4':
                ref_labels[i] = ref_labels[i].replace(':4', ':sus4')
            elif ref_labels[i][-2:] == ':6':
                ref_labels[i] = ref_labels[i].replace(':6', ':maj6')
            elif ref_labels[i][-4:] == ':6/2':
                ref_labels[i] = ref_labels[i].replace(':6/2', ':maj6/2')
            elif ref_labels[i] == 'Emin/4':
                ref_labels[i] = 'E:min/4'
            elif ref_labels[i] == 'A7/3':
                ref_labels[i] = 'A:7/3'
            elif ref_labels[i] == 'Bb7/3':
                ref_labels[i] = 'Bb:7/3'
            elif ref_labels[i] == 'Bb7/5':
                ref_labels[i] = 'Bb:7/5'
            elif ref_labels[i].find(':') == -1:
                if ref_labels[i].find('min') != -1:
                    ref_labels[i] = ref_labels[i][:ref_labels[i].find('min')] + ':' + ref_labels[i][ref_labels[i].find('min'):]
        return ref_labels

