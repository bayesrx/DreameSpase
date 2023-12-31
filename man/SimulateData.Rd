% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/SimulateData.R
\name{SimulateData}
\alias{SimulateData}
\title{Simulate data from the model on a rectangular lattice.}
\usage{
SimulateData(W, r, alpha, X, psi_2, Sigma, tau_2, rho, nu_2)
}
\arguments{
\item{W}{Adjacency matrix}

\item{r}{Number of rows in the lattice}

\item{alpha}{p x 1 vector of mean effects of covariates}

\item{X}{p x 1 vector of covariate values}

\item{psi_2}{Variance component of spatial effects CAR (eta)}

\item{tau_2}{Variance component of spatial error CAR (delta)}

\item{rho}{Correlation component of spatial error CAR (delta)}

\item{nu_2}{Variance of pure error}

\item{sigma_2}{variance of means conditional upon alpha}

\item{phi}{Correlation component of spatial effects CAR (eta)}
}
\value{
List
}
\description{
Simulate data from the model on a rectangular lattice.
}
