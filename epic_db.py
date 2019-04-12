import torch
import numpy as np
import json
from pathlib import Path
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import epic_kitchens.meta as metadata
from epic_kitchens.dataset.epic_dataset import EpicVideoDataset, GulpVideoSegment

class EPIC_Dataset(Dataset):

    #---------------------------------------------------------------------------------------------------
    '''
    dataset: Either train , test_seen, test_unseen
    '''
    def __init__(self, db_root_dir, dataset='train', with_metadata=True, nb_keyframes=2):
        #prepare directories
        self.class_type     = 'verb+noun' if dataset == 'train' else None
        self.db_root_dir    = Path(db_root_dir)        
        starter_kit_dir     = self.db_root_dir / 'starter-kit-action-recognition/'
        gulp_root           = starter_kit_dir / f'data/processed/gulp/rgb_{dataset}'       
        self.annotation_dir = self.db_root_dir / 'annotations/'  
        self.data_dir       = './input_data/'

        #prepare annotation files
        self.verb_dict        = metadata.verb_classes()
        self.noun_dict        = metadata.noun_classes()
        self.state_dict       = json.load(open(self.data_dir + 'state_mapping_v3.json', 'r'))['states_id']
        self.action2v_n       = json.load(open(self.data_dir + f'dict_actions.json'))
        self.v_n2action       = dict(zip(self.action2v_n.values(), self.action2v_n.keys()))
        self.videos_info      = metadata.video_info()
        self.many_verb_list   = sorted(list(metadata.many_shot_verbs()) + [len(self.verb_dict)])
        self.many_noun_list   = sorted(list(metadata.many_shot_nouns()) + [len(self.noun_dict)])
        self.many_action_dict = sorted([int(self.v_n2action[f'{v}_{n}']) for (v, n) in metadata.many_shot_actions()])
        self.nb_keyframes     = nb_keyframes

        #dataset properties
        self.n_verbs      = len(self.verb_dict)  
        self.n_nouns      = len(self.noun_dict) 
        self.n_states     = len(self.state_dict) 
        self.n_many_verbs = len(self.many_verb_list)  
        self.n_many_nouns = len(self.many_noun_list) 
        self.n_actions    = len(self.action2v_n)
        self.n_many_actions = len(self.many_action_dict)
        self.full_img_size  = (448, 448)

        self.noun_ignore_cls = list({x for x in range(0, self.n_nouns)}.difference(metadata.many_shot_nouns()))
        self.verb_ignore_cls = list({x for x in range(0, self.n_verbs)}.difference(metadata.many_shot_verbs()))
        self.action_ignore_cls = list({x for x in range(0, self.n_actions)}.difference(self.many_action_dict))
        
        # class_type value choices: ['verb', 'noun', 'verb+noun', None], None is used for the test sets, where there are no labels.
        self.gulp_db        = EpicVideoDataset(gulp_root, self.class_type, with_metadata=with_metadata)
        self.video_segments = sorted(self.gulp_db.video_segments, key= lambda x: x.id)
        if dataset == 'train':
            self.object_table    = metadata.training_object_labels()
            self.state_table     = self.get_object_states(self.nb_keyframes)

        #data transformations
        self.img_transfrom = transforms.Compose([
                                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0),
                                    transforms.Resize(self.full_img_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.show_img_transfrom = transforms.Compose([
                                    transforms.Resize((200, 200)),
                                    transforms.ToTensor()
                                    ])
    #---------------------------------------------------------------------------------------------------

    def __getitem__(self, index):

        video_segment = self.video_segments[index]
        keyframe_ids  = self.get_random_keyframes_ids(video_segment, self.nb_keyframes) 
        
        rgb_frames_lst     = self.gulp_db.load_frames(video_segment, indices=keyframe_ids)
        fovea_lst = np.ones((self.nb_keyframes, 4), dtype=int) * [0, 0, self.full_img_size[0], self.full_img_size[1]]
                
        sample = {
                'uid'             : video_segment.uid,
                'rgb_frame'       : torch.stack([self.img_transfrom(rgb_frame)[:, fovea[0]:fovea[2], fovea[1]:fovea[3]] 
                                                    for rgb_frame, fovea in zip(rgb_frames_lst, fovea_lst)], dim=0), 
                'keyframe_ids'    : torch.Tensor(keyframe_ids)
                } 
        
        #labels available
        if self.class_type is not None: 
            verb_class     = video_segment.metadata['verb_class']
            noun_class     = video_segment.metadata['noun_class']
            all_nouns_list = video_segment.metadata['all_noun_classes']
            all_nouns_many = self.to_categorial_multilabel([self.noun_to_many_shot(n) for n in all_nouns_list], n_classes=self.n_many_nouns) #one more class for few_nouns
            all_nouns      = self.to_categorial_multilabel(all_nouns_list, self.n_nouns)
            all_action     = int(self.v_n2action[f'{verb_class}_{noun_class}'])
            epic_objects_states = self.state_table[index]

            sample.update({
                'verb_class_many' : self.verb_to_many_shot(verb_class),
                'all_nouns_many'  : all_nouns_many,
                'all_nouns'       : all_nouns,
                'verb_class'      : verb_class, 
                'noun_class'      : noun_class,
                'action_class'    : all_action,
                'obj_states'      : epic_objects_states
            })
        return sample       

    #---------------------------------------------------------------------------------------------------
    
    def noun_to_many_shot(self, noun_class):
        return self.many_noun_list.index(noun_class) if metadata.is_many_shot_noun(noun_class) else self.n_many_nouns-1
    def verb_to_many_shot(self, verb_class):
        return self.many_verb_list.index(verb_class) if metadata.is_many_shot_verb(verb_class) else self.n_many_verbs-1

     #---------------------------------------------------------------------------------------------------
    def get_keyframes_ids(self, video_segment, nb_keyframes):
        seg_factors  = [t/(nb_keyframes-1) for t in range(0, nb_keyframes)] if nb_keyframes > 1 else [t for t in range(0, nb_keyframes)]
        keyframe_ids = [int((video_segment.num_frames -1) * f) for f in seg_factors]
        return keyframe_ids
    
    def get_random_keyframes_ids(self, video_segment, nb_keyframes):
        nb_splits = nb_keyframes + 1
        seg_factors  = [t/(nb_splits-1) for t in range(0, nb_splits)] if nb_splits > 1 else [t for t in range(0, nb_splits)]
        video_splits = [int((video_segment.num_frames -1) * f) for f in seg_factors]
        keyframe_ids = np.zeros(nb_keyframes, dtype=np.int)
        for s in range(0, nb_keyframes):
            keyframe_ids[s] = np.random.randint(video_splits[s], video_splits[s+1]) 
        return keyframe_ids

    #---------------------------------------------------------------------------------------------------

    def __len__(self):
        return len(self.video_segments)

    #---------------------------------------------------------------------------------------------------

    def to_categorial_multilabel(self, labels, n_classes):
        categorials = torch.zeros(n_classes)
        categorials[labels] = 1
        return categorials

    #---------------------------------------------------------------------------------------------------
    def frame_state_percentage(self, frame_id, num_frames, state_type):
        state_changing_frame_nb = 0.5*num_frames
        if frame_id < state_changing_frame_nb and state_type=='prestate':
            return  1-(frame_id/state_changing_frame_nb) 
        elif frame_id > state_changing_frame_nb and state_type=='poststate':
            return  frame_id/state_changing_frame_nb
        return 0.0 

    #---------------------------------------------------------------------------------------------------

    def get_object_states(self, nb_keyframes):
        state_dict_file = json.load(open(self.data_dir + 'state_mapping_v3.json', 'r'))
        verb2state_dict = state_dict_file['state_transitions']
        split_state     = int(np.floor(nb_keyframes/2))
        states_arr      = torch.zeros((len(self.video_segments), nb_keyframes, self.n_states))

        for i, segment in enumerate(self.video_segments):
            seg_v_class = str(segment.verb_class)
            state_dict  = verb2state_dict[seg_v_class] if seg_v_class in verb2state_dict else []
            
            for s in state_dict:
                condition   = any(hint in s['hints']        for hint in segment.narration.split(' ') + [segment.noun]) or s['hints'] == []
                freeze_from = any(word in segment.narration for word in ['still', 'continue', 'continuing', 'stay'])
                if condition:
                    if not freeze_from:
                        for kf in range(nb_keyframes):
                            states_arr[i, kf, s['from']] = self.frame_state_percentage(kf, segment.num_frames, 'prestate')
                            states_arr[i, kf, s['to']]   = self.frame_state_percentage(kf, segment.num_frames, 'poststate')
                    else:
                        for kf in range(nb_keyframes):
                            states_arr[i, kf, s['to']] = max(self.frame_state_percentage(kf, segment.num_frames, 'poststate') + 0.5, 1)
        return states_arr
    #---------------------------------------------------------------------------------------------------
