#' Generate an adjacency matrix for an m x n lattice
#' @param m number of rows in the lattice
#' @param n number of columns in the lattice
#' @return (m\*n) x (m\*n) adjacency matrix for an m x n lattice
RectangularAdjMat = function(m, n){
  adj_mat = matrix(0, nrow = m*n, ncol = m*n)
  for(r in 0:(nrow(adj_mat) - 1)){
    r_1 = r %% m
    c_1 = floor(r / m)
    for(c in r:(ncol(adj_mat) - 1)){
      r_2 = c %% m
      c_2 = floor(c / m)
      adj_mat[r + 1,c + 1] = (abs(r_1 - r_2) == 1) * (c_2 == c_1) +
        (abs(c_1 - c_2) == 1) * (r_2 == r_1)
    }
  }
  adj_mat[lower.tri(adj_mat)] = t(adj_mat)[lower.tri(adj_mat)]
  return(adj_mat)
}
