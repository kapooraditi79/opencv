# file 8
from typing import Dict
from typing import Optional
import math
import sys
sys.path.append(r'D:\Antares\plain2D')
for typing import dict, list, set, tuple, Optional
from collections import defaultdict
import numpy as np

class PersistentGroupTracker:
    """
    Assign stable identities to groups across frames.
    
    DBSCAN gives temporary observations
    Phase B converts them into persistent group identities
    """

    def __init__(self, fps: float, 
                jaccard_threshold: float = 0.5,
                min_active_threshold: float=2.0,
                max_inactive_threshold: float=2.0,
                grace_display_second: float= 1.0,
                start_group_id: int=100):
        
        # convert seconds to frames. 
        self.fps= fps
        self.jaccard_threshold= jaccard_threshold
        self.min_active_frames= min_active_threshold
        self.max_active_threshhold= max_inactive_threshold
        self.grace_display_seconds= grace_display_second
        self.start_group_id= start_group_id

        # core state
        # defining groups
        # initial states
        self.groups: dict[int, dict]={} # group_id-> grp_record
        self.next_group_id = start_group_id

        self.palette = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
            (255, 128, 0),  # Orange
            (128, 0, 255),  # Purple
            (0, 255, 128),  # Spring
            (255, 0, 128),  # Rose
        ]

    def _get_helper_groups(self)-> Dict[int, dict]:
        """
        returning the active groups that are okay fpr matching
        """

        return {
            gid: group for gid,group in self.groups.items() if group['active']==True
        }
    
    def _get_active_confirmed_groups(self)->Dict[int,dict]:
        """
        return only active and confirmed ones. 
        """
        return {
            gid:group for gid, group in self.groups.items() if group['active']==True and group['confirmed']==True
        }


