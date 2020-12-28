data {
  // number of customers
  int<lower=0> N;
  // observation period
  vector<lower=1>[N] T;
  // time between last transaction and observation period end
  vector<lower=0>[N] recency;
  // number of transactions
  vector<lower=1>[N] frequency;
}

parameters {
  // per-customer churn probability
  vector<lower=0, upper=1>[N] p;
  // per-customer transaction rate
  vector<lower=0>[N] lambda;

  // churn probability shape parameters
  real<lower=0> alpha;
  real<lower=0> beta;

  real<lower=0> lambda_shape;
  real<lower=0> lambda_rate;
}

model {
  // cache calculations
  vector[N] non_churn_prob = 1 - p;

  p ~ beta(alpha, beta);
  lambda ~ gamma(lambda_shape, lambda_rate);

  // increase log likelihood
  target += frequency .* log(lambda) + (frequency - 1) *. log(non_churn_prob);
  target += log(non_churn_prob .* exp(- lambda .* T) + p .* exp(- lambda .* recency));
}
