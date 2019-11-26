import tensorflow as tf

class tensorcox:
    '''
    Counting process representation of Coxs' Partial likelihood in Tensorflow

    Args:
        surv_ (:obj:`numpy array` of :obj:`float`): Survival object (start, stop, event)
        predictor_ (:obj:`numpy array` of :obj:`float`): Covariate effect (Nx1)
    '''
    def __init__(self, surv_, predictor_):
        self.surv_ = tf.convert_to_tensor(surv_, dtype=tf.float64)
        self.predictor_ = tf.convert_to_tensor(predictor_, dtype=tf.float64)
        self.exp_predictor_ = tf.exp(self.predictor_)
        self.events = self.surv_[:, 2, None]
        self.event_times = tf.boolean_mask(self.surv_[:, 1, None], tf.equal(self.events, 1), name='event_times')[:, None]

        if self.surv_.shape[1] != 3:
            raise ValueError('Surv object has to have 3 dimensions [start, stop, event]')


    def loglikelihood(self):
        '''
        Log-Likelihood for Cox model

        Returns:
            float: Loglikelihood value
        '''

        with tf.name_scope("risk_set"):
            r1 = tf.greater_equal(self.surv_[:, 1], self.event_times, name='r1')
            r2 = tf.less(self.surv_[:, 0], self.event_times, name='r2')
            risk_set = tf.cast(tf.logical_and(r1, r2), tf.float64, name='risk_set')
        with tf.name_scope("loglikelihood"):
            l1 = tf.reduce_sum(tf.multiply(self.predictor_, self.events), name='nominator')
            l2 = tf.reduce_sum(tf.log(tf.matmul(risk_set, self.exp_predictor_)), name='denominator')
            ll = tf.subtract(l1, l2, name='neg_loglikelihood')
        return ll

    def concordance(self, return_pairs=False, parallel_iterations=100):
        '''
        Harrels C-index

        Args:
            return_pairs (bool): Indicator to return indivudal pairs rather then concordance estiamte
            parallel_iterations (int): Number of parralel processes

        Returns:
            float - tuple: Concordance measure (Harrels' C)

            Disconcordant pairs, Concordanct pairs, Ties in predictor, Ties in Time
        '''

        # small offset for censored data
        surv_end = self.surv_[:, 1, None] + (0.00001 * ((self.events-1)*(-1)))

        # intervall survivl end and hazard for events
        hi = tf.boolean_mask(self.predictor_, tf.equal(self.events, 1))[:, None]

        # initalization - first observation
        ii_0 = tf.constant(1)
        cond1_0 = tf.greater_equal(surv_end[:, 0], self.event_times[0])
        cond2_0 = tf.less(self.surv_[:, 0], self.event_times[0])
        hazard_set_0 = tf.multiply(tf.cast(tf.logical_and(cond1_0, cond2_0), tf.float64), self.predictor_[:, 0])
        concordant_pairs_0 = tf.cast(tf.constant(0), tf.float64)[None]
        disconcordant_pairs_0 = tf.cast(tf.constant(0), tf.float64)[None]
        ties_x_0 = tf.cast(tf.constant(0), tf.float64)[None]
        ties_y_0 = tf.cast(tf.constant(0), tf.float64)[None]

        # extracting concordanct and disconcordant pairs for each event time
        cond_loop = lambda ii, hazard_set, concordant_pairs, disconcordant_pairs, ties_x, ties_y: ii < tf.cast(tf.reduce_sum(self.events), tf.int32)
        func = lambda ii, hazard_set, concordant_pairs, disconcordant_pairs, ties_x, ties_y: [ii+1,
        tf.multiply(tf.cast(tf.logical_and(tf.greater_equal(surv_end[:, 0], self.event_times[ii]), tf.less(self.surv_[:, 0], self.event_times[ii])), tf.float64), self.predictor_[:, 0]),
        concordant_pairs + tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_and(tf.less(hi[ii-1], hazard_set), tf.not_equal(hazard_set, 0)), tf.not_equal(self.event_times[ii-1], surv_end)[:,0]), tf.float64))[None],
        disconcordant_pairs + tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_and(tf.greater(hi[ii-1], hazard_set), tf.not_equal(hazard_set, 0)), tf.not_equal(self.event_times[ii-1], surv_end)[:,0]), tf.float64))[None],
        ties_x + tf.maximum(tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(hi[ii-1], hazard_set), tf.not_equal(self.event_times[ii-1], surv_end)[:,0]), tf.float64)) -1, 0)[None],
        ties_y + tf.maximum(tf.reduce_sum(tf.cast(tf.equal(self.event_times[ii-1], surv_end), tf.float64)) -1, 0)[None]]
        ii, hazard_set, concordant_pairs, disconcordant_pairs, ties_x, ties_y= tf.while_loop(
            cond_loop, func, loop_vars=[ii_0, hazard_set_0, concordant_pairs_0, disconcordant_pairs_0, ties_x_0, ties_y_0], parallel_iterations=parallel_iterations)
        # repeat for last obs
        hazard_set = tf.multiply(tf.cast(tf.logical_and(tf.greater_equal(surv_end[:, 0], self.event_times[-1]), tf.less(self.surv_[:, 0], self.event_times[-1])), tf.float64), self.predictor_[:, 0])
        concordant_pairs += tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_and(tf.less(hi[-1], hazard_set), tf.not_equal(hazard_set, 0)), tf.not_equal(self.event_times[-1], surv_end)[:,0]), tf.float64))[None]
        disconcordant_pairs += tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_and(tf.greater(hi[-1], hazard_set), tf.not_equal(hazard_set, 0)), tf.not_equal(self.event_times[-1], surv_end)[:,0]), tf.float64))[None]
        ties_x += tf.maximum(tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(hi[-1], hazard_set), tf.not_equal(self.event_times[-1], surv_end)[:,0]), tf.float64)) -1, 0)[None]
        ties_y += tf.maximum(tf.reduce_sum(tf.cast(tf.equal(self.event_times[-1], surv_end), tf.float64)) -1, 0)[None]
        ties_x = tf.divide(ties_x, 2)
        ties_y = tf.divide(ties_y, 2)
        if return_pairs:
            return (disconcordant_pairs, concordant_pairs, ties_y, ties_x)
        else:
            return((1 - ((concordant_pairs-disconcordant_pairs) / (concordant_pairs+disconcordant_pairs+ties_x)+1)/2))

    def baseline_hazard(self, predictor, parallel_iterations=100):
        '''
        Breslow estimator for baseline hazard

        Args:
            predictor_ (:obj:`numpy array` of :obj:`float`): Predictions from fitted model (Nx1)
            parallel_iterations (int): Number of parralel processes

        Returns:
            tuple: times, baseline_estimates

        '''
        predictor =  tf.convert_to_tensor(predictor, dtype=tf.float64)
        order = tf.nn.top_k(-self.event_times[:, 0], tf.shape(self.event_times)[0])[1]
        self.event_times = tf.gather(self.event_times, order)
        i0 = tf.constant(0)
        a0 =  tf.cast(tf.constant(0), tf.float64)[None]
        cond = lambda i, aa: i < tf.shape(self.event_times)[0]
        funct = lambda i, aa : [i+1, tf.concat([aa,  tf.divide(1, tf.reduce_sum(tf.multiply(tf.cast(tf.logical_and(tf.greater_equal(self.surv_[:, 1], self.event_times[i]), tf.less(self.surv_[:, 0], self.event_times[i])), tf.float64), tf.exp(predictor - tf.reduce_mean(predictor))[:, 0])))[None]], axis=0)]
        i, aa = tf.while_loop(
            cond, funct, loop_vars=[i0, a0], shape_invariants=[i0.get_shape(), tf.TensorShape([None])], parallel_iterations=parallel_iterations)
        return(self.event_times, aa[1:])
