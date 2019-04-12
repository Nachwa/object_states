import torch.nn.functional as F
import torch
import torch.nn as nn

class action_noun_state_cam(nn.Module):
    def __init__(self, many_noun_size, state_size, many_verb_size, nb_keyframes= 5): 
        super().__init__()

        self.nb_keyframes   = nb_keyframes
        self.verb_size      = 125
        self.noun_size      = 352
        self.state_size     = state_size
        self.many_verb_size = many_verb_size
        self.many_noun_size = many_noun_size
        self.action_size    = 2513
        self.mi_feat_size   = 820 #~nb many shot actions
        self.nb_vgg_size    = 512

        self.features_layers = torch.load('./input_data/vgg19_bn')[:-7] #relu included

        self.epic_shared_cam = nn.Conv2d(in_channels=self.nb_vgg_size, 
                                         out_channels=self.mi_feat_size,
                                         kernel_size=(3,3))

        self.epic_noun_cam   = nn.Conv2d(in_channels=self.mi_feat_size, 
                                         out_channels=self.many_noun_size,
                                         kernel_size=(1,1))
        self.epic_noun_gap   = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.all_noun_fc     = nn.Linear(self.many_noun_size, self.noun_size)

        self.epic_state_cam  = nn.Conv2d(in_channels=self.mi_feat_size,
                                         out_channels=self.state_size,
                                         kernel_size=(1,1))
        self.epic_state_gap  = nn.AdaptiveAvgPool2d(output_size=(1,1))

        self.epic_temp_verb = nn.Conv2d(in_channels=self.nb_keyframes, 
                                         out_channels=2,
                                         kernel_size=(1,1))
        self.epic_temp_noun = nn.Conv2d(in_channels=self.nb_keyframes, 
                                         out_channels=1,
                                         kernel_size=(1,1))


        self.many_verb_fc = nn.Linear(self.state_size*2, self.many_verb_size)
        self.verb_fc      = nn.Linear(self.many_verb_size, self.verb_size)
        self.action_fc    = nn.Linear(self.verb_size+self.noun_size, self.action_size)

        self.init_weights(self.epic_shared_cam)
        self.init_weights(self.epic_noun_cam)
        self.init_weights(self.epic_state_cam)
        self.init_weights(self.epic_temp_verb)
        self.init_weights(self.all_noun_fc)
        self.init_weights(self.verb_fc)

    def forward(self, input):

        all_nouns_features, all_state_list = [], []
        noun_many_list, all_nouns_list = [], []
        for frame in range(self.nb_keyframes):
            input_frame      = input[:, frame].view(-1, 3, input.shape[3], input.shape[4])
            vgg_features     = self.features_layers(input_frame) 
            
            epic_shared_maps = F.relu(self.epic_shared_cam(vgg_features))
            
            epic_nouns_maps  = self.epic_noun_cam(epic_shared_maps)
            epic_many_nouns  = self.epic_noun_gap(F.relu(epic_nouns_maps))
            epic_all_nouns  = self.all_noun_fc(epic_many_nouns.view(-1, self.many_noun_size))

            epic_state_maps  = self.epic_state_cam(epic_shared_maps)
            epic_many_state  = self.epic_state_gap(F.relu(epic_state_maps))
            
            noun_many_list.append(epic_many_nouns.view(-1, self.many_noun_size))
            all_nouns_list.append(epic_all_nouns.view( -1, self.noun_size))
            all_state_list.append(epic_many_state.view(-1, self.state_size))

        #select one verb for the set of keyframes (in temporal dimension) 
        verb_features = torch.stack(all_state_list, dim=1).view(-1, self.nb_keyframes, self.state_size, 1)
        verb_features = self.epic_temp_verb(verb_features).view(-1, self.state_size*2)
        #select one noun for the set of keyframes (in temporal dimension)
        noun_features = torch.stack(all_nouns_list, dim=1).view(-1, self.nb_keyframes, self.noun_size, 1)
        noun_features = self.epic_temp_noun(noun_features).view(-1, self.noun_size)

        many_verb = self.many_verb_fc(F.relu(verb_features))
        all_verbs = self.verb_fc(F.relu(many_verb))
        all_actions = self.action_fc(torch.cat([all_verbs.view(-1, self.verb_size),
                                                noun_features.view(-1, self.noun_size)], dim=1))

        return  (torch.stack(all_nouns_list, dim=1), torch.stack(noun_many_list, dim=1)), \
                (many_verb.view(-1, self.many_verb_size), torch.stack(all_state_list, dim=1)), \
                (all_verbs.view(-1, self.verb_size), noun_features.view(-1, self.noun_size)), \
                (all_actions.view(-1, self.action_size), None)

    def init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
        if type(m) == nn.Conv2d:
            nn.init.kaiming_normal_(m.weight)
            m.bias.data.zero_()


