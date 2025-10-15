#    Copyright 2024 Flash-VStream Authors 
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import json
import logging
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# from sklearn.cluster import DBSCAN
# from sklearn.decomposition import PCA
# from sklearn.mixture import GaussianMixture

import numpy as np

def drop_feature(img_feature, video_max_frames, img_similarity=None):
    T, P, D = img_feature.shape
    indices = [[i] for i in range(T)]
    T0 = video_max_frames
    if T <= T0:
        return img_feature, img_similarity, [indices]
    cur_feature = img_feature[:T0]  # [T0, P, D]
    if img_similarity is not None:
        cur_sim = img_similarity[:T0 - 1]
    else:
        cur_sim = F.cosine_similarity(cur_feature[:-1].view(T0 - 1, P * D), cur_feature[1:].view(T0 - 1, P * D))  # [T0 - 1]
    cur_indices = indices[:T0]
    step_indices = [cur_indices]
    for i in range(T0, T):
        new_feature = img_feature[i]
        new_sim = F.cosine_similarity(cur_feature[-1].view(-1), new_feature.view(-1), dim=0)
        all_feature = torch.cat([cur_feature, new_feature.unsqueeze(0)], dim=0)
        all_indices = cur_indices + [[i]]
        all_sim = torch.cat([cur_sim, new_sim.unsqueeze(0)], dim=0)
        idx = torch.argmax(all_sim)
        if random.randint(0, 1) > 0:
            idx = idx + 1
        cur_feature = torch.cat([all_feature[:idx], all_feature[idx + 1:]])
        if idx + 1 == T0 + 1:
            cur_sim = all_sim[:T0 - 1]
            cur_indices = all_indices[:-1] 
        elif idx == 0:
            cur_sim = all_sim[1:]
            cur_indices = all_indices[1:] 
        else:
            cur_sim = torch.cat([all_sim[:idx], all_sim[idx + 1:]])
            cur_sim[idx - 1] = F.cosine_similarity(all_feature[idx - 1].view(-1), all_feature[idx + 1].view(-1), dim=0)
            cur_indices = all_indices[:idx] + all_indices[idx + 1:]
        step_indices.append(cur_indices)
    print(f'Note: perform drop feature {img_feature.shape} to {cur_feature.shape}')
    return cur_feature, cur_sim, step_indices


def merge_feature(img_feature, video_max_frames, img_similarity=None):
    T, P, D = img_feature.shape
    indices = [[i] for i in range(T)]
    T0 = video_max_frames
    if T <= T0:
        return img_feature, img_similarity, [indices]
    cur_feature = img_feature[:T0]  # [T0, P, D]
    cur_indices = indices[:T0]
    step_indices = [cur_indices]
    if img_similarity is not None:
        cur_sim = img_similarity[:T0 - 1]
    else:
        cur_sim = F.cosine_similarity(cur_feature[:-1].view(T0 - 1, P * D), cur_feature[1:].view(T0 - 1, P * D))  # [T0 - 1]
    for i in range(T0, T):
        new_feature = img_feature[i]
        new_sim = F.cosine_similarity(cur_feature[-1].view(-1), new_feature.view(-1), dim=0)
        all_feature = torch.cat([cur_feature, new_feature.unsqueeze(0)], dim=0)
        all_sim = torch.cat([cur_sim, new_sim.unsqueeze(0)], dim=0)
        all_indices = cur_indices + [[i]]
        idx = torch.argmax(all_sim)
        all_feature[idx + 1] = (all_feature[idx] + all_feature[idx + 1]) / 2.0
        all_indices[idx + 1] = all_indices[idx] + all_indices[idx + 1]
        cur_feature = torch.cat([all_feature[:idx], all_feature[idx + 1:]])
        cur_sim = torch.cat([all_sim[:idx], all_sim[idx + 1:]])
        cur_indices = all_indices[:idx] + all_indices[idx + 1:]
        if idx > 0:
            cur_sim[idx - 1] = F.cosine_similarity(all_feature[idx - 1].view(-1), all_feature[idx + 1].view(-1), dim=0)
        if idx + 1 < T0:
            cur_sim[idx] = F.cosine_similarity(all_feature[idx + 1].view(-1), all_feature[idx + 2].view(-1), dim=0)
        step_indices.append(cur_indices)
    print(f'Note: perform merge feature {img_feature.shape} to {cur_feature.shape}')
    return cur_feature, cur_sim, step_indices


def kmeans_feature(img_feature, video_max_frames, img_similarity=None):
    def kmeans_torch(X, num_clusters, distance='euclidean', tol=1e-4, max_iter=10):
        indices = torch.randperm(X.size(0))[:num_clusters]
        centroids = X[indices]
        for i in range(max_iter):
            if distance == 'euclidean':
                dists = torch.cdist(X, centroids, p=2)
            else:
                raise NotImplementedError("Only Euclidean distance is supported yet")
            labels = torch.argmin(dists, dim=1)
            new_centroids = []
            for j in range(num_clusters):
                cluster_points = X[labels == j]
                if len(cluster_points) > 0:
                    new_centroid = cluster_points.mean(0)
                else:  # fix nan centroids
                    new_centroid = X[random.randint(0, X.size(0) - 1)]
                new_centroids.append(new_centroid)
            new_centroids = torch.stack(new_centroids)
            diff = torch.norm(centroids - new_centroids, dim=1).sum()
            if diff < tol:
                break
            centroids = new_centroids
        return centroids, labels, i
    T, P, D = img_feature.shape
    T0 = video_max_frames
    if T <= T0:
        return img_feature, img_similarity, [[[i] for i in range(T)]]
    X = img_feature.view(T, -1)  # [T, P, D]
    centroids, labels, exit_step = kmeans_torch(X, T0)
    reduced_feature = centroids.view(T0, P, D)
    # print(f'Note: perform kmeans feature {img_feature.shape} to {reduced_feature.shape}, exit at step={exit_step}')  # actually, K=T0
    step_indices = [[] for _ in range(T0)]
    for i in range(T0):
        step_indices[i] = [j for j in range(T) if labels[j] == i]
    return reduced_feature, img_similarity, [step_indices]


def weighted_kmeans_feature(img_feature, video_max_frames, weights=None):
    if weights is None:
        weights = torch.ones(img_feature.size(0), dtype=img_feature.dtype, device=img_feature.device)
    def weighted_kmeans_torch(X, num_clusters, weights=None, distance='euclidean', tol=1e-4, max_iter=10):
        indices = torch.randperm(X.size(0), device=X.device)[:num_clusters]
        centroids = X[indices]
        for i in range(max_iter):
            if distance == 'euclidean':
                dists = ((X.unsqueeze(1) - centroids.unsqueeze(0)) ** 2).sum(dim=2).sqrt()
            else:
                raise NotImplementedError("Only Euclidean distance is supported yet")
            labels = torch.argmin(dists, dim=1)
            weighted_sum = torch.zeros_like(centroids)
            weights_sum = torch.zeros(num_clusters, dtype=X.dtype, device=X.device)
            for j in range(num_clusters):
                cluster_mask = labels == j
                weighted_sum[j] = torch.sum(weights[cluster_mask, None] * X[cluster_mask], dim=0)
                weights_sum[j] = torch.sum(weights[cluster_mask])
            mask = weights_sum > 0
            new_centroids = torch.zeros_like(weighted_sum)
            new_centroids[mask] = weighted_sum[mask] / weights_sum[mask, None]
            if mask.sum() < num_clusters:  # fix nan centroids
                new_centroids[~mask] = torch.stack([X[random.randint(0, X.size(0) - 1)] for _ in range(num_clusters - mask.sum())])
            diff = torch.norm(centroids - new_centroids, dim=1).sum()
            if diff < tol:
                break
            centroids = new_centroids
        return centroids, labels, weights_sum, i
    T, P, D = img_feature.shape
    T0 = video_max_frames
    if T <= T0:
        return img_feature, weights, [[[i] for i in range(T)]]
    X = img_feature.view(T, -1)  # [T, P, D]
    centroids, labels, weights, exit_step = weighted_kmeans_torch(X, T0, weights)
    reduced_feature = centroids.view(T0, P, D)
    print(f'Note: perform weighted kmeans feature {img_feature.shape} to {reduced_feature.shape}, exit at step={exit_step}')  # actually, K=T0
    step_indices = [[] for _ in range(T0)]
    for i in range(T0):
        step_indices[i] = [j for j in range(T) if labels[j] == i]
    return reduced_feature, weights, [step_indices]


def weighted_kmeans_ordered_feature(img_feature, video_max_frames, weights=None, times=None):
    dtype = img_feature.dtype
    img_feature = img_feature.float()
    torch.set_printoptions(edgeitems=20, linewidth=180)

    if weights is None:
        weights = torch.ones(img_feature.size(0), dtype=img_feature.dtype, device=img_feature.device)
    if times is None:
        times = torch.arange(img_feature.size(0), dtype=img_feature.dtype, device=img_feature.device)
    # print(f'[{img_feature.device}]Note: start weighted kmeans ordered img_feature={img_feature.shape}, video_max_frames={video_max_frames}, weights={weights.shape}, times={times.shape}')  # actually, K=T0
    def efficient_euclidean_distance(A, B):
        print(f'efficient_euclidean_distance: A is on {A.device}, B is on {B.device}')
        assert A.ndim == 2
        assert B.ndim == 2
        assert A.shape[1] == B.shape[1]
        A_2 = torch.sum(A ** 2, dim=1, keepdim=True)
        B_2 = torch.sum(B ** 2, dim=1, keepdim=True)
        AB = A @ B.T
        dists_2 = A_2 + B_2.T - 2 * AB
        dists = torch.sqrt(dists_2)
        return dists
    def weighted_kmeans_torch(X, num_clusters, weights=None, distance='euclidean', tol=1e-4, max_iter=10):
        unique_X = torch.unique(X, dim=0)
        if unique_X.size(0) < num_clusters:
            print("Note Number of unique points is less than the number of clusters")
            centroids = unique_X
            if distance == 'euclidean':
                # dists = ((X.unsqueeze(1) - centroids.unsqueeze(0)) ** 2).sum(dim=2).sqrt()
                dists = efficient_euclidean_distance(X, centroids)
            else:
                raise NotImplementedError("Only Euclidean distance is supported yet")
            labels = torch.argmin(dists, dim=1)
            weights = torch.ones(centroids.size(0), dtype=centroids.dtype, device=centroids.device)
            return centroids, labels, weights, -1

        indices = torch.randperm(unique_X.size(0), device=X.device)[:num_clusters]
        centroids = unique_X[indices]

        for i in range(max_iter):
            if distance == 'euclidean':
                # dists = ((X.unsqueeze(1) - centroids.unsqueeze(0)) ** 2).sum(dim=2).sqrt()
                dists = efficient_euclidean_distance(X, centroids)
            else:
                raise NotImplementedError("Only Euclidean distance is supported yet")
            labels = torch.argmin(dists, dim=1)
            weighted_sum = torch.zeros_like(centroids)
            weights_sum = torch.zeros(num_clusters, dtype=X.dtype, device=X.device)
            for j in range(num_clusters):
                cluster_mask = labels == j
                weighted_sum[j] = torch.sum(weights[cluster_mask, None] * X[cluster_mask], dim=0)
                weights_sum[j] = torch.sum(weights[cluster_mask])
            mask = weights_sum > 0
            new_centroids = torch.zeros_like(weighted_sum)
            new_centroids[mask] = weighted_sum[mask] / weights_sum[mask, None]
            if mask.sum() < num_clusters:  # fix nan centroids
                for j in range(len(X)):
                    print(f'X[{j}]={X[j]}')
                new_centroids[~mask] = torch.stack([X[random.randint(0, X.size(0) - 1)] for _ in range(num_clusters - mask.sum())])
            diff = torch.norm(centroids - new_centroids, dim=1).sum()
            if diff < tol:
                break
            centroids = new_centroids
        return centroids, labels, weights_sum, i
    time_0 = time.perf_counter()
    T, P, D = img_feature.shape
    T0 = video_max_frames
    if T <= T0:
        return img_feature, weights, [[[i] for i in range(T)]]
    X = img_feature.view(T, -1)  # [T, P, D]
    centroids, labels, weights_sum, exit_step = weighted_kmeans_torch(X, T0, weights)
    time_1 = time.perf_counter()
    reduced_feature = centroids.view(-1, P, D)
    # print(f'[{img_feature.device}]Note: perform weighted kmeans ordered feature {img_feature.shape} to {reduced_feature.shape}, exit at step={exit_step}')  # actually, K=T0
    # step_indices, centroids_timestamp = [], []
    # for i in range(reduced_feature.shape[0]):
    #     indices, timestamp, total_weight = [], 0, 0
    #     for j in range(T):
    #         if labels[j] == i:
    #             indices.append(j)
    #             timestamp = timestamp + times[j] * weights[j]
    #             total_weight = total_weight + weights[j]
    #     step_indices.append(indices)
    #     timestamp = timestamp / total_weight
    #     centroids_timestamp.append(timestamp)
    step_indices = [[] for _ in range(reduced_feature.shape[0])]
    centroids_timestamp = torch.zeros(reduced_feature.shape[0], dtype=times.dtype, device=times.device)
    total_weights = torch.zeros(reduced_feature.shape[0], dtype=weights.dtype, device=weights.device)
    for j in range(T):
        cluster_id = labels[j]
        step_indices[cluster_id].append(j)
        centroids_timestamp[cluster_id] += times[j] * weights[j]
        total_weights[cluster_id] += weights[j]
    for i in range(reduced_feature.shape[0]):
        if total_weights[i] > 0:
            centroids_timestamp[i] /= total_weights[i]
    time_2 = time.perf_counter()
        
    centroids_timestamp = [sum(indices) / len(indices) for indices in step_indices]
    centroids_timestamp = torch.tensor(centroids_timestamp, device=img_feature.device)
    # print(f'[{centroids.device}]Note, centroids_timestamp={centroids_timestamp}')
    sorted_indices = torch.argsort(centroids_timestamp)  # from less to more
    sorted_reduced_feature = reduced_feature[sorted_indices]
    sorted_weights = weights_sum[sorted_indices]
    centroids_timestamp = centroids_timestamp[sorted_indices]
    sorted_step_indices = [step_indices[i] for i in sorted_indices]
    time_3 = time.perf_counter()
    if exit_step == -1:
        print(f'Note: {sorted_reduced_feature.shape} is less than {T0}, cat some features')
        pad_len = T0 - sorted_reduced_feature.shape[0]
        sorted_reduced_feature = torch.cat([img_feature[:pad_len], sorted_reduced_feature])
        sorted_weights = torch.cat([torch.ones(pad_len, device=img_feature.device), sorted_weights])
        centroids_timestamp = torch.cat([torch.arange(pad_len, device=img_feature.device), centroids_timestamp])
        sorted_step_indices = [[i] for i in list(range(pad_len))] + sorted_step_indices
    time_4 = time.perf_counter()
    time_list = [time_1 - time_0, time_2 - time_1, time_3 - time_2, time_4 - time_3]
    logger = logging.getLogger(__name__ + '.weighted_kmeans_ordered_feature')
    logger.info(f'Note: time_list={time_list}')
    return sorted_reduced_feature.to(dtype), sorted_weights, centroids_timestamp, sorted_step_indices


def fast_weighted_kmeans_ordered_feature(img_feature, video_max_frames, weights=None):
    dtype = img_feature.dtype
    img_feature = img_feature.float()
    torch.set_printoptions(edgeitems=20, linewidth=180)

    if weights is None:
        weights = torch.ones(img_feature.size(0), dtype=img_feature.dtype, device=img_feature.device)
    def weighted_kmeans_torch(X, num_clusters, weights=None, distance='euclidean', tol=1e-4, max_iter=10):
        unique_X = torch.unique(X, dim=0)
        if unique_X.size(0) < num_clusters:
            print("Note Number of unique points is less than the number of clusters")
            centroids = unique_X
            if distance == 'euclidean':
                # dists = ((X.unsqueeze(1) - centroids.unsqueeze(0)) ** 2).sum(dim=2).sqrt()
                A_2 = torch.sum(X ** 2, dim=1, keepdim=True)
                B_2 = torch.sum(centroids ** 2, dim=1, keepdim=True)
                AB = X @ centroids.T
                dists_2 = A_2 + B_2.T - 2 * AB
                dists = torch.sqrt(dists_2)
            else:
                raise NotImplementedError("Only Euclidean distance is supported yet")
            labels = torch.argmin(dists, dim=1)
            weights = torch.ones(centroids.size(0), dtype=centroids.dtype, device=centroids.device)
            return centroids, labels, weights, -1

        indices = torch.randperm(unique_X.size(0), device=X.device)[:num_clusters]
        centroids = unique_X[indices]

        for i in range(max_iter):
            if distance == 'euclidean':
                # dists = ((X.unsqueeze(1) - centroids.unsqueeze(0)) ** 2).sum(dim=2).sqrt()
                A_2 = torch.sum(X ** 2, dim=1, keepdim=True)
                B_2 = torch.sum(centroids ** 2, dim=1, keepdim=True)
                AB = X @ centroids.T
                dists_2 = A_2 + B_2.T - 2 * AB
                dists = torch.sqrt(dists_2)
            else:
                raise NotImplementedError("Only Euclidean distance is supported yet")
            labels = torch.argmin(dists, dim=1)  # [T], (0~T0-1)
            weighted_sum = torch.zeros_like(centroids)
            weights_sum = torch.zeros(num_clusters, dtype=X.dtype, device=X.device)
            for j in range(num_clusters):
                cluster_mask = labels == j
                weighted_sum[j] = torch.sum(weights[cluster_mask, None] * X[cluster_mask], dim=0)
                weights_sum[j] = torch.sum(weights[cluster_mask])
            mask = weights_sum > 0
            new_centroids = torch.zeros_like(weighted_sum)
            new_centroids[mask] = weighted_sum[mask] / weights_sum[mask, None]
            if mask.sum() < num_clusters:  # fix nan centroids
                for j in range(len(X)):
                    print(f'X[{j}]={X[j]}')
                new_centroids[~mask] = torch.stack([X[random.randint(0, X.size(0) - 1)] for _ in range(num_clusters - mask.sum())])
            diff = torch.norm(centroids - new_centroids, dim=1).sum()
            if diff < tol:
                break
            centroids = new_centroids
        return centroids, labels, weights_sum, i
    T, P, D = img_feature.shape
    T0 = video_max_frames
    if T <= T0:
        return img_feature, weights, [[[i] for i in range(T)]]
    X = img_feature.view(T, -1)  # [T, P*D], T -> T0
    centroids, labels, weights, exit_step = weighted_kmeans_torch(X, T0, weights)  # [T]
    reduced_feature = centroids.view(-1, P, D)
    print(f'[{img_feature.device}]Note: perform weighted kmeans ordered feature {img_feature.shape} to {reduced_feature.shape}, exit at step={exit_step}')  # actually, K=T0
    step_indices = [[] for _ in range(reduced_feature.shape[0])]

    for i in range(reduced_feature.shape[0]):
        step_indices[i] = [j for j in range(T) if labels[j] == i]
    centroids_timestamp = [sum(indices) / len(indices) for indices in step_indices]
    centroids_timestamp = torch.tensor(centroids_timestamp, device=img_feature.device)
    # print(f'[{centroids.device}]Note, centroids_timestamp={centroids_timestamp}')
    sorted_indices = torch.argsort(torch.tensor(centroids_timestamp))  # from less to more
    sorted_reduced_feature = reduced_feature[sorted_indices]
    sorted_weights = weights[sorted_indices]
    centroids_timestamp = centroids_timestamp[sorted_indices]
    sorted_step_indices = [step_indices[i] for i in sorted_indices]
    if exit_step == -1:
        print(f'Note: {sorted_reduced_feature.shape} is less than {T0}, cat some features')
        pad_len = T0 - sorted_reduced_feature.shape[0]
        sorted_reduced_feature = torch.cat([img_feature[:pad_len], sorted_reduced_feature])
        sorted_weights = torch.cat([torch.ones(pad_len, device=img_feature.device), sorted_weights])
        centroids_timestamp = torch.cat([torch.arange(pad_len, device=img_feature.device), centroids_timestamp])
        sorted_step_indices = [[i] for i in list(range(pad_len))] + sorted_step_indices
    return sorted_reduced_feature.to(dtype), sorted_weights, centroids_timestamp, sorted_step_indices


def pca_weighted_kmeans_ordered_feature(img_feature, video_max_frames, weights=None):
    dtype = img_feature.dtype
    img_feature = img_feature.float()
    torch.set_printoptions(edgeitems=20, linewidth=180)

    if weights is None:
        weights = torch.ones(img_feature.size(0), dtype=img_feature.dtype, device=img_feature.device)
    def weighted_kmeans_torch(X, num_clusters, weights=None, distance='euclidean', tol=1e-4, max_iter=10):
        unique_X = torch.unique(X, dim=0)
        if unique_X.size(0) < num_clusters:
            print("Note Number of unique points is less than the number of clusters")
            centroids = unique_X
            if distance == 'euclidean':
                dists = ((X.unsqueeze(1) - centroids.unsqueeze(0)) ** 2).sum(dim=2).sqrt()
            else:
                raise NotImplementedError("Only Euclidean distance is supported yet")
            labels = torch.argmin(dists, dim=1)
            weights = torch.ones(centroids.size(0), dtype=centroids.dtype, device=centroids.device)
            return centroids, labels, weights, -1

        indices = torch.randperm(unique_X.size(0), device=X.device)[:num_clusters]
        centroids = unique_X[indices]

        for i in range(max_iter):
            if distance == 'euclidean':
                dists = ((X.unsqueeze(1) - centroids.unsqueeze(0)) ** 2).sum(dim=2).sqrt()
            else:
                raise NotImplementedError("Only Euclidean distance is supported yet")
            labels = torch.argmin(dists, dim=1)
            weighted_sum = torch.zeros_like(centroids)
            weights_sum = torch.zeros(num_clusters, dtype=X.dtype, device=X.device)
            for j in range(num_clusters):
                cluster_mask = labels == j
                weighted_sum[j] = torch.sum(weights[cluster_mask, None] * X[cluster_mask], dim=0)
                weights_sum[j] = torch.sum(weights[cluster_mask])
            mask = weights_sum > 0
            new_centroids = torch.zeros_like(weighted_sum)
            new_centroids[mask] = weighted_sum[mask] / weights_sum[mask, None]
            if mask.sum() < num_clusters:  # fix nan centroids
                for j in range(len(X)):
                    print(f'X[{j}]={X[j]}')
                new_centroids[~mask] = torch.stack([X[random.randint(0, X.size(0) - 1)] for _ in range(num_clusters - mask.sum())])
            diff = torch.norm(centroids - new_centroids, dim=1).sum()
            if diff < tol:
                break
            centroids = new_centroids
        return centroids, labels, weights_sum, i
    T, P, D = img_feature.shape
    T0 = video_max_frames
    if T <= T0:
        return img_feature, weights, [[[i] for i in range(T)]]
    
    pca_dim = 32
    img_feature_reshaped = img_feature.view(T * P, D)
    pca = PCA(n_components=pca_dim)
    img_feature_pca_reduced = pca.fit_transform(img_feature_reshaped.detach().cpu().numpy())
    img_feature_pca_reduced = torch.tensor(img_feature_pca_reduced, device=img_feature.device).view(T * P, pca_dim)
    
    X = img_feature_pca_reduced.view(T, -1)  # [T, P * pca_dim]
    centroids_pca_reduced, labels, weights, exit_step = weighted_kmeans_torch(X, T0, weights)  # [T, P * pca_dim]
    num_clusters = centroids_pca_reduced.shape[0]
    # reduced_feature = centroids.view(-1, P, D)
    labels_one_hot = F.one_hot(labels, num_classes=num_clusters).float()  # [T, T0]
    cluster_counts = labels_one_hot.sum(dim=0)  # [T0]
    cluster_counts[cluster_counts == 0] = 1  # 避免除以零
    reduced_feature = torch.einsum('tk,tpd->kpd', labels_one_hot, img_feature)
    reduced_feature = reduced_feature / cluster_counts.unsqueeze(-1).unsqueeze(-1)  # [T0, P, D]

    print(f'[{img_feature.device}]Note: perform PCA-weighted kmeans ordered feature {img_feature.shape} to {reduced_feature.shape}, exit at step={exit_step}')  # actually, K=T0
    step_indices = [[] for _ in range(reduced_feature.shape[0])]

    for i in range(reduced_feature.shape[0]):
        step_indices[i] = [j for j in range(T) if labels[j] == i]
    centroids_timestamp = [sum(indices) / len(indices) for indices in step_indices]
    centroids_timestamp = torch.tensor(centroids_timestamp, device=img_feature.device)
    # print(f'[{centroids.device}]Note, centroids_timestamp={centroids_timestamp}')
    sorted_indices = torch.argsort(torch.tensor(centroids_timestamp))  # from less to more
    sorted_reduced_feature = reduced_feature[sorted_indices]
    sorted_weights = weights[sorted_indices]
    centroids_timestamp = centroids_timestamp[sorted_indices]
    sorted_step_indices = [step_indices[i] for i in sorted_indices]
    if exit_step == -1:
        print(f'Note: {sorted_reduced_feature.shape} is less than {T0}, cat some features')
        pad_len = T0 - sorted_reduced_feature.shape[0]
        sorted_reduced_feature = torch.cat([img_feature[:pad_len], sorted_reduced_feature])
        sorted_weights = torch.cat([torch.ones(pad_len, device=img_feature.device), sorted_weights])
        centroids_timestamp = torch.cat([torch.arange(pad_len, device=img_feature.device), centroids_timestamp])
        sorted_step_indices = [[i] for i in list(range(pad_len))] + sorted_step_indices
    return sorted_reduced_feature.to(dtype), sorted_weights, centroids_timestamp, sorted_step_indices


def torchpca_weighted_kmeans_ordered_feature(img_feature, video_max_frames, weights=None, pca_dim=32):
    dtype = img_feature.dtype
    img_feature = img_feature.float()
    torch.set_printoptions(edgeitems=20, linewidth=180)

    if weights is None:
        weights = torch.ones(img_feature.size(0), dtype=img_feature.dtype, device=img_feature.device)
    def pca_torch(X, k):  # [N, D], k -> [N, k]
        X_mean = torch.mean(X, dim=0)
        X_centered = X - X_mean
        # 计算协方差矩阵
        covariance_matrix = torch.mm(X_centered.T, X_centered) / (X_centered.size(0) - 1)
        # 计算协方差矩阵的特征值和特征向量
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)
        # 选择前 k 个特征向量
        topk_eigenvectors = eigenvectors[:, :k]
        # 投影数据到前 k 个主成分上
        X_pca = torch.mm(X_centered, topk_eigenvectors)
        return X_pca

    def weighted_kmeans_torch(X, num_clusters, weights=None, distance='euclidean', tol=1e-4, max_iter=10):
        unique_X = torch.unique(X, dim=0)
        if unique_X.size(0) < num_clusters:
            print("Note Number of unique points is less than the number of clusters")
            centroids = unique_X
            if distance == 'euclidean':
                dists = ((X.unsqueeze(1) - centroids.unsqueeze(0)) ** 2).sum(dim=2).sqrt()
            else:
                raise NotImplementedError("Only Euclidean distance is supported yet")
            labels = torch.argmin(dists, dim=1)
            weights = torch.ones(centroids.size(0), dtype=centroids.dtype, device=centroids.device)
            return centroids, labels, weights, -1

        indices = torch.randperm(unique_X.size(0), device=X.device)[:num_clusters]
        centroids = unique_X[indices]

        for i in range(max_iter):
            if distance == 'euclidean':
                dists = ((X.unsqueeze(1) - centroids.unsqueeze(0)) ** 2).sum(dim=2).sqrt()
            else:
                raise NotImplementedError("Only Euclidean distance is supported yet")
            labels = torch.argmin(dists, dim=1)
            weighted_sum = torch.zeros_like(centroids)
            weights_sum = torch.zeros(num_clusters, dtype=X.dtype, device=X.device)
            for j in range(num_clusters):
                cluster_mask = labels == j
                weighted_sum[j] = torch.sum(weights[cluster_mask, None] * X[cluster_mask], dim=0)
                weights_sum[j] = torch.sum(weights[cluster_mask])
            mask = weights_sum > 0
            new_centroids = torch.zeros_like(weighted_sum)
            new_centroids[mask] = weighted_sum[mask] / weights_sum[mask, None]
            if mask.sum() < num_clusters:  # fix nan centroids
                for j in range(len(X)):
                    print(f'X[{j}]={X[j]}')
                new_centroids[~mask] = torch.stack([X[random.randint(0, X.size(0) - 1)] for _ in range(num_clusters - mask.sum())])
            diff = torch.norm(centroids - new_centroids, dim=1).sum()
            if diff < tol:
                break
            centroids = new_centroids
        return centroids, labels, weights_sum, i
    T, P, D = img_feature.shape
    T0 = video_max_frames
    if T <= T0:
        return img_feature, weights, [[[i] for i in range(T)]]
    
    img_feature_reshaped = img_feature.view(T * P, D)
    img_feature_pca_reduced = pca_torch(img_feature_reshaped, pca_dim) # [T * P, pca_dim]
    
    X = img_feature_pca_reduced.view(T, -1)  # [T, P * pca_dim]
    centroids_pca_reduced, labels, weights, exit_step = weighted_kmeans_torch(X, T0, weights)  # [T, P * pca_dim]
    num_clusters = centroids_pca_reduced.shape[0]
    # reduced_feature = centroids.view(-1, P, D)
    labels_one_hot = F.one_hot(labels, num_classes=num_clusters).float()  # [T, T0]
    cluster_counts = labels_one_hot.sum(dim=0)  # [T0]
    cluster_counts[cluster_counts == 0] = 1  # 避免除以零
    reduced_feature = torch.einsum('tk,tpd->kpd', labels_one_hot, img_feature)
    reduced_feature = reduced_feature / cluster_counts.unsqueeze(-1).unsqueeze(-1)  # [T0, P, D]

    print(f'[{img_feature.device}]Note: perform PCA-weighted kmeans ordered feature {img_feature.shape} to {reduced_feature.shape}, exit at step={exit_step}')  # actually, K=T0
    step_indices = [[] for _ in range(reduced_feature.shape[0])]

    for i in range(reduced_feature.shape[0]):
        step_indices[i] = [j for j in range(T) if labels[j] == i]
    centroids_timestamp = [sum(indices) / len(indices) for indices in step_indices]
    centroids_timestamp = torch.tensor(centroids_timestamp, device=img_feature.device)
    # print(f'[{centroids.device}]Note, centroids_timestamp={centroids_timestamp}')
    sorted_indices = torch.argsort(torch.tensor(centroids_timestamp))  # from less to more
    sorted_reduced_feature = reduced_feature[sorted_indices]
    sorted_weights = weights[sorted_indices]
    centroids_timestamp = centroids_timestamp[sorted_indices]
    sorted_step_indices = [step_indices[i] for i in sorted_indices]
    if exit_step == -1:
        print(f'Note: {sorted_reduced_feature.shape} is less than {T0}, cat some features')
        pad_len = T0 - sorted_reduced_feature.shape[0]
        sorted_reduced_feature = torch.cat([img_feature[:pad_len], sorted_reduced_feature])
        sorted_weights = torch.cat([torch.ones(pad_len, device=img_feature.device), sorted_weights])
        centroids_timestamp = torch.cat([torch.arange(pad_len, device=img_feature.device), centroids_timestamp])
        sorted_step_indices = [[i] for i in list(range(pad_len))] + sorted_step_indices
    return sorted_reduced_feature.to(dtype), sorted_weights, centroids_timestamp, sorted_step_indices


def k_drop_feature(img_feature, video_max_frames, img_similarity=None):
    T, P, D = img_feature.shape
    indices = [[i] for i in range(T)]
    T0 = video_max_frames
    if T <= T0:
        return img_feature, img_similarity, [indices]
    cur_feature = img_feature[:T0]  # [T0, P, D]
    normed_cur_features = F.normalize(cur_feature.view(T0, P * D), p=2, dim=1)
    cur_sim = torch.mm(normed_cur_features, normed_cur_features.T)  # [T0, T0]
    cur_sim.fill_diagonal_(-100.0)
    cur_indices = indices[:T0]
    step_indices = [cur_indices]
    for i in range(T0, T):
        # get new feature
        new_feature = img_feature[i]
        normed_new_feature = F.normalize(new_feature.view(1, P * D), p=2, dim=1)
        new_sim = torch.mm(normed_cur_features, normed_new_feature.T)  # [T0, 1]
        all_feature = torch.cat([cur_feature, new_feature.unsqueeze(0)], dim=0)
        normed_all_features = torch.cat([normed_cur_features, normed_new_feature], dim=0)
        all_indices = cur_indices + [[i]]
        # get new similarity
        all_sim_1 = torch.cat([cur_sim, new_sim], dim=1)  # [T0, T0 + 1]
        all_sim = torch.cat([all_sim_1, torch.ones_like(all_sim_1[-1:]) * -100.0], dim=0)  # [T0 + 1, T0 + 1]
        all_sim[-1, :-1] = new_sim.T
        # choose compression position
        idx = torch.argmax(all_sim)
        left, right = idx // (T0 + 1), idx % (T0 + 1)
        if random.randint(0, 1) > 0:
            idx = left
        else:
            idx = right
        assert all_sim[left, right] == torch.max(all_sim)
        # get compressed feature and similarity
        cur_feature = torch.cat([all_feature[:idx], all_feature[idx + 1:]])
        normed_cur_features = torch.cat([normed_all_features[:idx], normed_all_features[idx + 1:]])
        cur_indices = all_indices[:idx] + all_indices[idx + 1:]
        cur_sim_1 = torch.cat([all_sim[:idx], all_sim[idx + 1:]], dim=0)  # [T0, T0 + 1]
        cur_sim = torch.cat([cur_sim_1[:, :idx], cur_sim_1[:, idx + 1:]], dim=1)  # [T0, T0]
        step_indices.append(cur_indices)
    # print(f'Note: perform k-drop feature {img_feature.shape} to {cur_feature.shape}')
    return cur_feature, None, step_indices


def k_merge_feature(img_feature, video_max_frames, img_similarity=None):
    T, P, D = img_feature.shape
    indices = [[i] for i in range(T)]
    T0 = video_max_frames
    if T <= T0:
        return img_feature, img_similarity, [indices]
    cur_feature = img_feature[:T0]  # [T0, P, D]
    normed_cur_features = F.normalize(cur_feature.view(T0, P * D), p=2, dim=1)
    cur_sim = torch.mm(normed_cur_features, normed_cur_features.T)  # [T0, T0]
    cur_sim.fill_diagonal_(-100.0)
    cur_indices = indices[:T0]
    step_indices = [cur_indices]
    for i in range(T0, T):
        # get new feature
        new_feature = img_feature[i]
        normed_new_feature = F.normalize(new_feature.view(1, P * D), p=2, dim=1)
        new_sim = torch.mm(normed_cur_features, normed_new_feature.T)  # [T0, 1]
        all_feature = torch.cat([cur_feature, new_feature.unsqueeze(0)], dim=0)
        normed_all_features = torch.cat([normed_cur_features, normed_new_feature], dim=0)
        all_indices = cur_indices + [[i]]
        # get new similarity
        all_sim_1 = torch.cat([cur_sim, new_sim], dim=1)  # [T0, T0 + 1]
        all_sim = torch.cat([all_sim_1, torch.ones_like(all_sim_1[-1:]) * -100.0], dim=0)  # [T0 + 1, T0 + 1]
        all_sim[-1, :-1] = new_sim.T
        # choose compression position
        idx = torch.argmax(all_sim)
        left, right = idx // (T0 + 1), idx % (T0 + 1)
        assert all_sim[left, right] == torch.max(all_sim)
        # update feature
        all_feature[right] = (all_feature[left] + all_feature[right]) / 2.0
        normed_all_features[right] = F.normalize(all_feature[right].view(1, P * D), p=2, dim=1)
        all_indices[right] = all_indices[left] + all_indices[right]
        # update similarity
        new_sim = torch.mm(normed_all_features, normed_all_features[right:right+1].T)  # [T0 + 1, 1]
        all_sim[right, :] = new_sim.T
        all_sim[:, right:right+1] = new_sim
        all_sim[right, right] = -100.0
        # get compressed feature and similarity
        cur_feature = torch.cat([all_feature[:left], all_feature[left + 1:]])
        normed_cur_features = torch.cat([normed_all_features[:left], normed_all_features[left + 1:]])
        cur_indices = all_indices[:left] + all_indices[left + 1:]
        cur_sim_1 = torch.cat([all_sim[:left], all_sim[left + 1:]], dim=0)  # [T0, T0 + 1]
        cur_sim = torch.cat([cur_sim_1[:, :left], cur_sim_1[:, left + 1:]], dim=1)  # [T0, T0]
        step_indices.append(cur_indices)
    # print(f'Note: perform k-merge feature {img_feature.shape} to {cur_feature.shape}')
    return cur_feature, cur_sim, step_indices


def dbscan_feature(img_feature, video_max_frames, img_similarity=None):
    T, P, D = img_feature.shape
    T0 = video_max_frames
    X = img_feature.reshape(T, -1).to(torch.float32).cpu().numpy()
    db = DBSCAN(eps=0.62, min_samples=2).fit(X)
    labels = db.labels_
    labels = list(labels)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = labels.count(-1)
    idx = n_clusters
    new_labels = []
    for l in labels:
        if l == -1:
            if idx < T0:
                new_labels.append(idx)
                idx += 1
            else:
                new_labels.append(-1)
        else:
            new_labels.append(l)
    n_clusters = len(set([x for x in new_labels if x!= -1]))
    cluster_features = []
    new_labels = torch.tensor(new_labels, device=img_feature.device)
    for i in range(n_clusters):
        cluster_features.append(img_feature[new_labels == i].mean(dim=0))
    cluster_features = torch.stack(cluster_features, dim=0)
    if n_clusters < T0:
        print(f'Note: cat {cluster_features.shape} to {T0} frames')
        cluster_features = torch.cat([img_feature[:T0 - n_clusters], cluster_features])
    print(f'Note: perform dbscan feature {img_feature.shape} to {cluster_features.shape}')
    return cluster_features, None, None


def gmm_feature(img_feature, video_max_frames, img_similarity=None):
    T, P, D = img_feature.shape
    T0 = video_max_frames
    X = img_feature.reshape(T, -1).float().cpu().numpy()
    pca = PCA(n_components=32)  # Adjust n_components as needed
    X = pca.fit_transform(X)
    gmm = GaussianMixture(n_components=T0, random_state=0)
    gmm.fit(X)
    cluster_labels = gmm.predict(X)
    cluster_labels = torch.tensor(cluster_labels, device=img_feature.device)
    cluster_features = []
    for i in range(T0):
        cluster_features.append(img_feature[cluster_labels == i].mean(dim=0))
    cluster_features = torch.stack(cluster_features, dim=0)
    cluster_features = cluster_features.to(img_feature.dtype)
    return cluster_features, None, None


def attention_feature(img_feature, video_max_frames, attention_fn=None):
    T, P, D = img_feature.shape
    T0 = video_max_frames
    # print(f'attention feature from {img_feature.shape} to {video_max_frames}')
    if T <= T0:
        return img_feature, None, [[[i] for i in range(T)]], None
    cur_feature = img_feature[:T0]  # [T0, P, D]
    turing_memory = cur_feature.reshape(T0*P, D)  # [T0*P, D], T0=4, n<=4
    
    for i in range(T0, T, T0):
        j = min(i + T0, T)
        new_feature = img_feature[i:j]  # [P, D]
        new_feature = new_feature.reshape(-1, D)  # [n*P, D]
        turing_memory = attention_fn(turing_memory, new_feature)  # [T0*P, n*P]

    cur_feature = turing_memory.reshape(T0, P, D)
    print(f'Note: perform {attention_fn.__name__} feature {img_feature.shape} to {cur_feature.shape}')
    return cur_feature, None, [[[i] for i in range(T0)]], None
