%% this script will compute the smaller eigenvectors of the laplacian.
% There was a need of rewriting this in Octave because scipy is very slow

DEADLINES = {
    "test",
    "2021-06-01",
    "2022-01-01",
    "2022-07-01",
    "2023-01-01",
    "2024-01-01",
};

NEIGS = 10;

for i = 1:rows(DEADLINES)
    fprintf("==== %d deadline %s\n", i, DEADLINES{i})
    fname = sprintf("./data/adj_%s.m.gz", DEADLINES{i});

    % read the adjacency matrix as a list of weighted edges (i, j, A_ij)
    M = load(fname);

    % increase indexing (Octave count from 1)
    M(:, 1) = M(:, 1) + 1;
    M(:, 2) = M(:, 2) + 1;

    % define a sparse matrix
    mat = spconvert(M);

    % compute the degree
    deg = sum(mat);
    dim = columns(mat);

    % laplacian
    lap = diag(deg) - mat;

    % compute the eigenpairs
    opt.tol=1e-9;
    opt.issym=true;
    [bevecs, bevals] = eigs(lap, NEIGS + 1, 'sm', opt);
    bevals = diag(bevals);

    % drop the Frobenious eigenvector
    bevecs = bevecs(:, 1:NEIGS);
    fname = sprintf("./data/embedding_laplacian_%s.txt.gz", DEADLINES{i});
    save("-ascii","-zip", fname, "bevecs");

    % normalized laplacian
    invdeg = 1.0 ./ sqrt(deg);
    normlap = diag(invdeg) * lap * diag(invdeg);

    % compute the eigenpairs
    % opt.tol=1e-9;
    opt.issym=true;
    [bevecs, bevals] = eigs(normlap, NEIGS + 1, 'sm', opt);
    bevals = diag(bevals);

    % drop the Frobenious eigenvector
    bevecs = bevecs(:, 1:NEIGS);
    fname = sprintf("./data/embedding_norm_laplacian_%s.txt.gz", DEADLINES{i});
    save("-ascii","-zip", fname, "bevecs");
end
