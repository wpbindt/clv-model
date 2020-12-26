data {
    int<lower=0> N; // number of customers
    vector<lower=0>[N] value; // mean repeated transaction value
    vector<lower=1>[N] frequency; // repeat transactions
}

parameters {
    real<lower=0> p; // global shape parameter
    vector<lower=0>[N] nu; // per-customer rate parameter
    real<lower=0> q; // shape parameter underlying rate parameter
    real<lower=0> mu; // rate parameter underlying rate parameter
}

model {
    nu ~ gamma(q, mu);
    value ~ gamma(frequency * p, frequency .* nu);
}
