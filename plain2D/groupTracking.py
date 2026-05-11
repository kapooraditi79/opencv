def jaccard_similarity(set1, set2):

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    if union == 0:
        return 0

    return intersection / union


def build_clusters(person_ids, labels):

    clusters = {}

    for idx, label in enumerate(labels):

        if label == -1:
            continue

        person_id = person_ids[idx]

        if label not in clusters:
            clusters[label] = set()

        clusters[label].add(person_id)

    return clusters

def match_groups(
    current_clusters,
    active_groups,
    frame_idx,
    next_group_id,
    threshold=0.5
):

    assigned_groups = {}

    used_old_groups = set()

    # =====================================================
    # match current clusters to old groups
    # =====================================================

    for cluster_members in current_clusters.values():

        best_match = None
        best_score = 0

        for group_id, group_data in active_groups.items():

            if group_id in used_old_groups:
                continue

            old_members = group_data['members']

            score = jaccard_similarity(
                cluster_members,
                old_members
            )

            if score > best_score:
                best_score = score
                best_match = group_id

        # =================================================
        # inherit old group
        # =================================================

        if best_score >= threshold:

            assigned_groups[best_match] = cluster_members

            used_old_groups.add(best_match)

        # =================================================
        # create new group
        # =================================================

        else:

            assigned_groups[next_group_id] = cluster_members

            next_group_id += 1

    # =====================================================
    # update active groups
    # =====================================================

    for group_id, members in assigned_groups.items():

        active_groups[group_id] = {
            'members': members,
            'last_seen': frame_idx
        }

    return assigned_groups, next_group_id