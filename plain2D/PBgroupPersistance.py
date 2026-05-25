# file 8

from sklearn import cluster
from typing import Dict, List, Set, Tuple, Optional
import math
import sys
sys.path.append(r'D:\Antares\plain2D')
from typing import Optional

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
                min_active_threshold: float=0.5,
                max_inactive_threshold: float=2.0,
                grace_display_second: float= 1.0,
                start_group_id: int=100,
                pending_inactive_threshold: float=1.5):
        
        # convert seconds to frames. 
        self.fps= fps
        self.jaccard_threshold= jaccard_threshold
        self.min_active_frames= int(min_active_threshold * fps)
        self.max_inactive_frames= int(max_inactive_threshold* fps)
        self.grace_display_frames= int(grace_display_second* fps)
        self.PENDING_COLOR = (128, 128, 128)
        self.pending_max_inactive_frames = int(pending_inactive_threshold * fps)
        self.debug= False


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

    def _get_eligible_groups(self)-> dict[int, dict]:
        """
        returning the active groups that are okay fpr matching
        """

        return {
            gid: group for gid,group in self.groups.items() if group['active']==True
        }
    
    def _get_active_confirmed_groups(self)->dict[int,dict]:
        """
        return only active and confirmed ones. 
        """
        return {
            gid:group for gid, group in self.groups.items() if group['active']==True and group['confirmed']==True
        }
    
    def _get_active_pending_groups(self) -> dict[int, dict]:
        """
        Return active but unconfirmed groups.
        """

        return {
            gid: group
            for gid, group in self.groups.items()
            if group["active"] and not group["confirmed"]
        }

    def _is_in_grace_period(self,group:dict, current_frame: int)->bool:
        """
        returns the fading groups. will have a fading color
        """
        if group['active']:
            return False
        if not group['confirmed']:
            return False
        
        frames_inactive= current_frame-group['last_seen']
        return frames_inactive<=(self.max_inactive_frames + self.grace_display_frames)

    def _build_clusters_from_labels(self, track_cluster_map:dict[int, int])-> list[set[int]]:
        """
        Convert DBSCAN labels to list of track ID sets.
        
        Args:
            track_cluster_map: {track_id: cluster_label}
                              cluster_label = -1 means noise
        
        Returns:
            List of sets, each set contains track_ids in that cluster
            Example: [{17, 23, 41}, {52, 68}] for clusters 0 and 1
        """
        clusters_dict= defaultdict(set)
        for track_id, label in track_cluster_map.items():
            if label==-1:
                continue
            clusters_dict[label].add(track_id)

        return list(clusters_dict.values())


    def _compute_jaccard(self, cluster:set[int], group_members:set[int])->float:
        """
        Compute Jaccard similarity between a cluster and a group.
            
            J(A,B) = |A ∩ B| / |A ∪ B|
        """ 
        if not cluster or not group_members:
            return 0.0

        intersection= len(cluster & group_members)
        union= len(cluster | group_members)

        return intersection/union if union>0 else 0.0

    def _match_clusters_to_groups(self, current_clusters: list[set[int]], eligible_groups: dict[int, dict]):
        """
            Greedy matching of clusters to groups using Jaccard similarity.
            
            Strategy:
            1. Compute Jaccard for every (cluster, group) pair
            2. Sort by score descending
            3. Greedy assignment: each cluster and group used at most once
            4. Only accept matches above threshold
            
            Args:
                current_clusters: List of track ID sets from current frame
                eligible_groups: {group_id: group_record} for active groups
            
            Returns:
                List of tuples [(cluster_index, group_id, jaccard_score), ...]
                Sorted by score descending (best matches first)
        """

        # compute the pairwise similarities first
        candidates= []
        for cluster_idx, cluster_mmebers in enumerate(current_clusters):
            for group_id, group in eligible_groups.items():
                score= self._compute_jaccard(cluster_mmebers,group['members'])

                if score>= self.jaccard_threshold:
                    candidates.append((score,cluster_idx,group_id))

            # sort by highest jaccard
        candidates.sort(key= lambda x : x[0], reverse= True)

            # greedy assignment
        matched_clusters = set()
        matched_groups = set()
        matches = []
    
        for score, cluster_idx, group_id in candidates:
            if cluster_idx in matched_clusters:
                continue  # This cluster already claimed
            if group_id in matched_groups:
                continue  # This group already claimed
        
            # Valid match
            matched_clusters.add(cluster_idx)
            matched_groups.add(group_id)
            matches.append((cluster_idx, group_id, score))
        
        return matches


    def _create_new_group(self, cluster_members: Set[int], current_frame: int) -> int:
        """
        Create a new pending group from an unmatched cluster.
        
        Returns:
            The new group_id
        """
        group_id = self.next_group_id
        self.next_group_id += 1
        
        self.groups[group_id] = {
            "group_id": group_id,
            "members": cluster_members.copy(),  # Current membership
            "created_at": current_frame,
            "first_confirmed_at": None,
            "last_seen": current_frame,
            "active": True,      # Eligible for matching next frame
            "confirmed": False,  # Pending until min_active met
            "color": None,       # Assigned only on confirmation
            "history": [(current_frame, len(cluster_members))]  # Start history
        }
        
        return group_id
    

    def _update_matched_group(self, 
                              group_id: int, 
                              cluster_members: Set[int], 
                              current_frame: int) -> dict:
        """
        Update a group that was matched to a current cluster.
        
        Returns:
            Dict with event info: {'type': 'confirmed' or 'updated', 'group_id': group_id}
        """
        group = self.groups[group_id]
        
        # Update membership and timestamp
        group["members"] = cluster_members.copy()
        group["last_seen"] = current_frame
        group["active"] = True  # Ensure active (might have been revived)
        
        # Add to history (only if confirmed)
        if group["confirmed"]:
            group["history"].append((current_frame, len(cluster_members)))
        
        event = {'group_id': group_id, 'type': 'updated'}
        
        # Check if pending group now meets confirmation threshold
        if not group["confirmed"]:
            frames_alive = current_frame - group["created_at"]+1
            
            if frames_alive >= self.min_active_frames:
                group["confirmed"] = True
                group["first_confirmed_at"] = current_frame
                group["color"] = self._assign_color(group_id, current_frame)
                event['type'] = 'confirmed'
                print(f"[Frame {current_frame}] Group {group_id} CONFIRMED! "
                      f"(alive for {frames_alive} frames, {len(cluster_members)} members)")
        
        return event

    def _assign_color(self, group_id: int, current_frame: int) -> Tuple[int, int, int]:
        """
        Assign a unique color to a newly confirmed group.
        
        Strategy:
        1. Get all colors currently used by active+grace groups
        2. Pick first palette color not in use
        3. If all colors in use, recycle the oldest active group's color
        
        Args:
            group_id: The group to assign color to
            current_frame: Current frame number (for grace period check)
        
        Returns:
            RGB tuple
        """
        # Get colors of all currently active or in-grace groups
        used_colors = set()
        
        for gid, group in self.groups.items():
            if group.get("color") is None:
                continue
            
            # Check if this group is active OR in grace period
            is_active = group["active"]
            # FIXED: Pass current_frame, not group["last_seen"]
            is_grace = (not is_active and 
                       group["confirmed"] and 
                       self._is_in_grace_period(group, current_frame))
            
            if is_active or is_grace:
                used_colors.add(group["color"])
        
        # Find first unused color
        for color in self.palette:
            if color not in used_colors:
                return color
        
        # All colors in use - recycle oldest active group's color
        # Find active group with smallest last_seen (oldest)
        recyclable_groups = [
        (gid, g)
        for gid, g in self.groups.items()
        if (
            not g["active"]
            and not self._is_in_grace_period(g, current_frame)
            and g.get("color") is not None
    )
]
        if recyclable_groups:
            oldest_gid = min(recyclable_groups, key=lambda x: x[1]["last_seen"])[0]
            recycled_color = self.groups[oldest_gid]["color"]
            print(f"[Warning] Color palette exhausted. Recycling color {recycled_color} "
                  f"from group {oldest_gid} for group {group_id}")
            return recycled_color
        
        # Fallback (should never happen)
        return self.palette[group_id % len(self.palette)]



    def _get_person_to_group_map(
        self,
        track_cluster_map: Dict[int, int],
        cluster_to_group: Dict[int, int],
        current_clusters: List[Set[int]]
    ) -> Dict[int, Optional[int]]:

        person_groups = {}

        # Default = ungrouped
        for track_id in track_cluster_map.keys():
            person_groups[track_id] = None

        # Assign every cluster's group_id
        for cluster_idx, group_id in cluster_to_group.items():

            if cluster_idx >= len(current_clusters):
                continue

            for track_id in current_clusters[cluster_idx]:
                group = self.groups[group_id]

                if group["confirmed"]:
                    person_groups[track_id] = group_id

        return person_groups
    
    def _deactivate_stale_groups(self,
                             matched_group_ids: Set[int],
                             current_frame: int) -> List[int]:

        deactivated = []

        eligible_groups = self._get_eligible_groups()

        for group_id, group in eligible_groups.items():

            if group_id in matched_group_ids:
                continue

            frames_missing = current_frame - group["last_seen"]

            # Pending groups die fast
            if not group["confirmed"]:
                threshold = self.pending_max_inactive_frames
            else:
                threshold = self.max_inactive_frames

            if frames_missing >= threshold:
                group["active"] = False

                deactivated.append(group_id)

                print(
                    f"[Frame {current_frame}] "
                    f"Group {group_id} DEACTIVATED "
                    f"(missing for {frames_missing} frames)"
                )

        return deactivated
    
    def _get_display_groups(self, current_frame: int) -> Dict:
        """
        Prepare group information for the display layer.
        
        Returns dict with:
            - active_groups: Confirmed + active (solid borders, full opacity)
            - grace_groups: Confirmed + inactive but within grace period (fading)
            - pending_groups: Not yet confirmed (dashed borders)
        """
        active_groups = {}
        grace_groups = {}
        pending_groups = {}
        
        for group_id, group in self.groups.items():
            # Skip historical groups (inactive and beyond grace period)
            if not group["active"] and not self._is_in_grace_period(group, current_frame):
                continue
            
            if not group["confirmed"]:
                # Pending group (active but not confirmed)
                if group["active"]:
                    pending_groups[group_id] = {
                        "members": group["members"],
                        "age_frames": current_frame - group["created_at"]+1,
                        "age_seconds": (current_frame - group["created_at"]+1) / self.fps,
                        "color": self.PENDING_COLOR
                    }
            
            elif group["active"]:
                # Active + Confirmed
                active_groups[group_id] = {
                    "members": group["members"],
                    "color": group["color"],
                    "confirmed": True,
                    "age_seconds": (current_frame - group["first_confirmed_at"]) / self.fps,
                    "member_count": len(group["members"])
                }
            
            elif self._is_in_grace_period(group, current_frame):
                # Inactive but in grace period (fading)
                frames_inactive = current_frame - group["last_seen"]
                grace_progress = frames_inactive - self.max_inactive_frames
                
                # Calculate opacity: 1.0 at start of grace, 0.0 at end
                opacity = 1.0 - (grace_progress / self.grace_display_frames)
                opacity = max(0.0, min(1.0, opacity))
                
                grace_groups[group_id] = {
                    "members": group["members"],
                    "color": group["color"],
                    "opacity": opacity,
                    "inactive_seconds": frames_inactive / self.fps
                }
        
        return {
            "active_groups": active_groups,
            "grace_groups": grace_groups,
            "pending_groups": pending_groups
        }


    def update(self, 
               track_cluster_map: Dict[int, int],
               track_boxes: Dict[int, List[float]],
               current_frame: int) -> Dict:
        """
        Main update method - processes one frame of data.
        
        Args:
            track_cluster_map: {track_id: cluster_label} from DBSCAN
            track_boxes: {track_id: [x1, y1, x2, y2]} for display
            current_frame: Current frame number
        
        Returns:
            Dictionary with:
                - person_groups: {track_id: group_id or None}
                - active_groups: Display info for active confirmed groups
                - grace_groups: Display info for fading groups
                - pending_groups: Display info for unconfirmed groups
                - events: List of events this frame
                - track_boxes: Original boxes (passthrough for display)
        """
        events = []
        
        # Step 1: Convert DBSCAN labels to cluster sets
        current_clusters = self._build_clusters_from_labels(track_cluster_map)
        
        # # Step 2: Get eligible groups for matching
        # eligible_groups = self._get_eligible_groups()
        
        # # Step 3: Match current clusters to existing groups
        # matches = self._match_clusters_to_groups(current_clusters, eligible_groups)

        # Stage 1: confirmed groups get priority
        confirmed_groups = self._get_active_confirmed_groups()

        confirmed_matches = self._match_clusters_to_groups(
            current_clusters,
            confirmed_groups
        )

        matched_cluster_indices = {m[0] for m in confirmed_matches}

        # Remaining unmatched clusters
        remaining_clusters = []
        remaining_cluster_map = {}

        for idx, cluster in enumerate(current_clusters):

            if idx not in matched_cluster_indices:

                remaining_cluster_map[len(remaining_clusters)] = idx
                remaining_clusters.append(cluster)

        # Stage 2: pending groups match remaining clusters
        pending_groups = self._get_active_pending_groups()

        pending_matches_local = self._match_clusters_to_groups(
            remaining_clusters,
            pending_groups
        )

        # Convert local indices back to original cluster indices
        pending_matches = []

        for local_idx, group_id, score in pending_matches_local:

            original_idx = remaining_cluster_map[local_idx]

            pending_matches.append(
                (original_idx, group_id, score)
            )

        # Final combined matches
        matches = confirmed_matches + pending_matches



        matched_cluster_indices = {m[0] for m in matches}
        matched_group_ids = {m[1] for m in matches}

        cluster_to_group = {}
        
        # Step 4: Update matched groups
        for cluster_idx, group_id, score in matches:
            cluster_to_group[cluster_idx] = group_id
            if cluster_idx < len(current_clusters):
                event = self._update_matched_group(
                    group_id, 
                    current_clusters[cluster_idx], 
                    current_frame
                )
                events.append(event)
        
        # Step 5: Create new groups for unmatched clusters
        for cluster_idx, cluster_members in enumerate(current_clusters):
            if cluster_idx not in matched_cluster_indices:
                new_group_id = self._create_new_group(cluster_members, current_frame)
                cluster_to_group[cluster_idx] = new_group_id
                events.append({
                    'type': 'created_pending',
                    'group_id': new_group_id,
                    'member_count': len(cluster_members)
                })
                print(f"[Frame {current_frame}] Created PENDING group {new_group_id} "
                      f"with {len(cluster_members)} members")
        
        # Step 6: Deactivate stale groups
        deactivated = self._deactivate_stale_groups(matched_group_ids, current_frame)
        for gid in deactivated:
            events.append({'type': 'deactivated', 'group_id': gid})
        
        # Step 7: Build return value for display
        person_groups = self._get_person_to_group_map(
            track_cluster_map, cluster_to_group, current_clusters
        )
        
        display_groups = self._get_display_groups(current_frame)
        
        # Step 8: Return complete state
        return {
            "person_groups": person_groups,
            "active_groups": display_groups["active_groups"],
            "grace_groups": display_groups["grace_groups"],
            "pending_groups": display_groups["pending_groups"],
            "events": events,
            "track_boxes": track_boxes  # Passthrough for display
        }
    
