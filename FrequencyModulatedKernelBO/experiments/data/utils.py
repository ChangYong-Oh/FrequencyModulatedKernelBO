from typing import Tuple

import torch

MAX_RANDOM_SEED = 2 ** 31 - 1


def generate_adjacency_matrix(graph_type: str, n_vertices: int) -> torch.Tensor:
    assert graph_type in ['path', 'complete', 'random']
    if graph_type == 'path':
        adjacency_matrix = torch.zeros(n_vertices, n_vertices)
        for i in range(n_vertices - 1):
            adjacency_matrix[i, i + 1] = 1
            adjacency_matrix[i + 1, i] = 1
    elif graph_type == 'complete':
        adjacency_matrix = torch.ones(n_vertices, n_vertices)
        for i in range(n_vertices):
            adjacency_matrix[i, i] = 0
    elif graph_type == 'random':
        adjacency_matrix = torch.zeros(n_vertices, n_vertices)
        for i in range(n_vertices):
            for j in range(i + 1, n_vertices):
                if torch.rand(1) > 0.5:
                    adjacency_matrix[i, j] = 1
                    adjacency_matrix[j, i] = 1
    else:
        raise NotImplementedError
    return adjacency_matrix


def graph_fourier(adjacency_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """

    """
    assert adjacency_matrix.dim() == 2
    assert torch.sum(torch.diag(adjacency_matrix) ** 2) == 0
    assert torch.sum((adjacency_matrix - adjacency_matrix.t()) ** 2) == 0
    degree_matrix = torch.diag(torch.sum(adjacency_matrix, dim=0))
    graph_laplacian = degree_matrix - adjacency_matrix
    fourier_frequency, fourier_basis = torch.symeig(graph_laplacian, eigenvectors=True)
    return fourier_frequency, fourier_basis


def path_graph_fourier(n_vertices: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    adjacency_matrix = generate_adjacency_matrix('path', n_vertices)
    fourier_frequency, fourier_basis = graph_fourier(adjacency_matrix)
    return adjacency_matrix, fourier_frequency, fourier_basis


def complete_graph_fourier(n_vertices: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    adjacency_matrix = generate_adjacency_matrix('complete', n_vertices)
    fourier_frequency, fourier_basis = graph_fourier(adjacency_matrix)
    return adjacency_matrix, fourier_frequency, fourier_basis


if __name__ == '__main__':
    _n_vertices = 8
    _beta = 5
    for _ in range(10):
        _adj_mat = generate_adjacency_matrix('random', _n_vertices)
        _laplacian = torch.diag(torch.sum(_adj_mat, dim=0)) - _adj_mat
        _eigval, _eigvec = torch.symeig(_laplacian, eigenvectors=True)
        print(torch.min(torch.matmul(_eigvec * torch.exp(-_beta * _eigval.clamp(min=0)), _eigvec.t())))