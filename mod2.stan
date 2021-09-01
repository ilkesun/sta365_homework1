data {
  int<lower=0> N;
  vector[N] y;
  vector<lower=-1, upper=2.5>[N] t;
  
  //priors
  real<lower=0> tau_a;
  real<lower=0> tau_b;
  real<lower=0> tau_s;

  int<lower=0, upper=1> only_prior;
}

parameters {
  real a;
  real<upper=0> b;
  real<lower=0> sigma;
}

transformed parameters {
  vector[N] log_mu = a + b * t;
}

model {
  //priors
  a ~ normal(0, tau_a);
  b ~ normal(0, tau_b);
  sigma ~ normal(0, tau_s);
  
  if(only_prior == 0) {
    y ~ normal(log_mu, sigma);
  }
}

generated quantities {
  vector[N] log_lik;
  vector[N] y_pred;
  for (i in 1:N) {
    log_lik[i] = lognormal_lpdf(y[i] | log_mu[i], sigma);
    y_pred[i] = normal_rng(log_mu[i], sigma);
  }
}
