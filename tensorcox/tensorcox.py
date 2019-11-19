class tensorcox:
    '''
    Survival analysis based on counting process representation of Coxs' Partial likelihood in Tensorflow
    '''
    def __init__(self, surv_, X_, theta_, linpred_=None, Z_=None, strata_=None):
        if np.any(strata_ == None):
            self.surv_ = surv_
        else:
            self.surv_ = surv_
            self.surv_[:, 0] = tf.multiply(self.surv_[:, 0], tf.multiply(strata_, 1000))
            self.surv_[:, 1] = tf.multiply(self.surv_[:, 1], tf.multiply(strata_, 1000))
        self.X_ = X_
        self.theta_ = theta_
        self.Z_ = Z_
        self.strata_ = strata_
        if np.any(linpred_ == None):
            if np.any(Z_ == None):
                self.linpred_ =tf.matmul(X_, theta_)
            else:
                self.linpred_ =tf.matmul(X_, tf.multiply(theta_, Z_))
        else:
            self.linpred_ = linpred_

    def negll(self):
        '''
        function that implements Coxs' partial likelihood based on a counting process representation
        - fully written in TF

        inputs:
        surv_ - TF arrary N x 3                  - survival object - [:,0] start-time - [:,1] end-time - [:,2] - event-indicator
        X_ - TF array N x 1                      - covariates
        theta_ - TF array N x 1                  - parameters
        Z_ - TF array N x _                      - design matrix for multilevel
        strata_ -  TF array N x 1                - array with strat assignment

        outputs:
        neg_ll TF array 1                        - partial likelihood estimate
        '''
        with tf.name_scope("risk_set"):
            events = self.surv_[:, 2, None]
            ti = tf.boolean_mask(self.surv_[:, 1, None], tf.equal(
                events, 1), name='event_times')[:, None]
            r1 = tf.greater_equal(self.surv_[:, 1], ti, name='r1')
            r2 = tf.less(self.surv_[:, 0], ti, name='r2')
            risk_set = tf.cast(tf.logical_and(r1, r2), tf.float64, name='risk_set')
        # likelihood
        with tf.name_scope("likelihood"):
            l1 = tf.reduce_sum(tf.multiply(self.linpred_, events), name='nominator')
            exp_linpred = tf.exp(self.linpred_, name='exp_linear_predictor')
            exp_linpred_riskset = tf.matmul(risk_set, exp_linpred)
            l2 = tf.reduce_sum(tf.log(exp_linpred_riskset), name='denominator')
            neg_ll = tf.subtract(l2, l1, name='neg_loglikelihood')
        return neg_ll

    def concordance(self, return_pairs=False):
        '''
        function that computes the concordance (similar to R - Survival package) for time varying covariates
        - fully written in TF

        inputs:
        surv - TF arrary N x 3           - survival object - [:,0] start-time - [:,1] end-time - [:,2] - event-indicator
        linpred - TF array N x 1         - predictor for covaraite influence on hazard
        return_pairs - bool                 - returns concordant/disconcordant ect. pairs for alternate concordance computation

        outputs:HYDR
        scalar:                             - concordance measure
        tuple:                              - pairs
        '''

        events = self.surv_[:, 2, None]
        surv_end = self.surv_[:, 1, None]
        surv_end = surv_end + (0.00001 * ((events-1)*(-1))) # small offset for censored data

        # intervall survivl end and hazard for events
        hi = tf.boolean_mask(self.linpred_, tf.equal(events, 1))[:, None]
        ti = tf.boolean_mask(surv_end, tf.equal(events, 1))[:, None]

        # initalization - first observation
        ii_0 = tf.constant(1)
        cond1_0 = tf.greater_equal(surv_end[:, 0], ti[0])
        cond2_0 = tf.less(self.surv_[:, 0], ti[0])
        hazard_set_0 = tf.multiply(tf.cast(tf.logical_and(cond1_0, cond2_0), tf.float64), self.linpred_[:, 0])
        concordant_pairs_0 = tf.cast(tf.constant(0), tf.float64)[None]
        disconcordant_pairs_0 = tf.cast(tf.constant(0), tf.float64)[None]
        ties_x_0 = tf.cast(tf.constant(0), tf.float64)[None]
        ties_y_0 = tf.cast(tf.constant(0), tf.float64)[None]

        # extracting concordanct and disconcordant pairs for each event time
        cond_loop = lambda ii, hazard_set, concordant_pairs, disconcordant_pairs, ties_x, ties_y: ii < tf.cast(tf.reduce_sum(events), tf.int32)
        func = lambda ii, hazard_set, concordant_pairs, disconcordant_pairs, ties_x, ties_y: [ii+1,
        tf.multiply(tf.cast(tf.logical_and(tf.greater_equal(surv_end[:, 0], ti[ii]), tf.less(self.surv_[:, 0], ti[ii])), tf.float64), self.linpred_[:, 0]),
        concordant_pairs + tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_and(tf.less(hi[ii-1], hazard_set), tf.not_equal(hazard_set, 0)), tf.not_equal(ti[ii-1], surv_end)[:,0]), tf.float64))[None],
        disconcordant_pairs + tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_and(tf.greater(hi[ii-1], hazard_set), tf.not_equal(hazard_set, 0)), tf.not_equal(ti[ii-1], surv_end)[:,0]), tf.float64))[None],
        ties_x + tf.maximum(tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(hi[ii-1], hazard_set), tf.not_equal(ti[ii-1], surv_end)[:,0]), tf.float64)) -1, 0)[None],
        ties_y + tf.maximum(tf.reduce_sum(tf.cast(tf.equal(ti[ii-1], surv_end), tf.float64)) -1, 0)[None]]
        ii, hazard_set, concordant_pairs, disconcordant_pairs, ties_x, ties_y= tf.while_loop(
            cond_loop, func, loop_vars=[ii_0, hazard_set_0, concordant_pairs_0, disconcordant_pairs_0, ties_x_0, ties_y_0])
        # repeat for last obs
        hazard_set = tf.multiply(tf.cast(tf.logical_and(tf.greater_equal(surv_end[:, 0], ti[-1]), tf.less(self.surv_[:, 0], ti[-1])), tf.float64), self.linpred_[:, 0])
        concordant_pairs += tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_and(tf.less(hi[-1], hazard_set), tf.not_equal(hazard_set, 0)), tf.not_equal(ti[-1], surv_end)[:,0]), tf.float64))[None]
        disconcordant_pairs += tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_and(tf.greater(hi[-1], hazard_set), tf.not_equal(hazard_set, 0)), tf.not_equal(ti[-1], surv_end)[:,0]), tf.float64))[None]
        ties_x += tf.maximum(tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(hi[-1], hazard_set), tf.not_equal(ti[-1], surv_end)[:,0]), tf.float64)) -1, 0)[None]
        ties_y += tf.maximum(tf.reduce_sum(tf.cast(tf.equal(ti[-1], surv_end), tf.float64)) -1, 0)[None]
        ties_x = tf.divide(ties_x, 2)
        ties_y = tf.divide(ties_y, 2)
        if return_pairs:
            return (disconcordant_pairs, concordant_pairs, ties_y, ties_x)
        else:
            return((1 - ((concordant_pairs-disconcordant_pairs) / (concordant_pairs+disconcordant_pairs+ties_x)+1)/2))

    def baseline_hazard(self, parallel_iterations=100):
        '''
        function to pbtain a estimator for the baseline hazard based on Breslows estimator
        '''
        self.linpred_ = self.linpred_ - tf.reduce_mean(self.linpred_)
        exp_linpred = tf.exp(self.linpred_)
        events = self.surv_[:, 2, None]
        ti = tf.boolean_mask(self.surv_[:, 1, None], tf.equal(events, 1), name='event_times')[:, None]
        order = tf.nn.top_k(-ti[:, 0], tf.shape(ti)[0])[1]
        ti = tf.gather(ti, order)
        i0 = tf.constant(0)
        a0 =  tf.cast(tf.constant(0), tf.float64)[None]
        cond = lambda i, aa: i < tf.shape(ti)[0]
        funct = lambda i, aa : [i+1, tf.concat([aa,  tf.divide(1, tf.reduce_sum(tf.multiply(tf.cast(tf.logical_and(tf.greater_equal(self.surv_[:, 1], ti[i]), tf.less(self.surv_[:, 0], ti[i])), tf.float64), exp_linpred[:, 0])))[None]], axis=0)]
        i, aa = tf.while_loop(
            cond, funct, loop_vars=[i0, a0], shape_invariants=[i0.get_shape(), tf.TensorShape([None])], parallel_iterations=1000)
        return(aa, ti)
