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
  vector<lower=0>[N] lambda; // per-customer transaction rate
  vector<lower=0>[N] mu; // per-customer churn rate

  // shape and rate underlying lambda
  real<lower=0> lambda_shape;
  real<lower=0> lambda_rate;

  // shape and rate underlying mu
  real<lower=0> mu_shape;
  real<lower=0> mu_rate;
}

model {
  // cache some calculations
  vector[N] lambda_plus_mu = lambda + mu;

  lambda ~ gamma(lambda_shape, lambda_rate);
  mu ~ gamma(mu_shape, mu_rate);

  // increment log likelihood
  target += frequency .* log(lambda) - log(lambda_plus_mu);
  target += log(mu .* exp(-lambda_plus_mu .* recency) + lambda .* exp(-lambda_plus_mu .* T));
}
