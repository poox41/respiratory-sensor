package com.example.breathheartdemo

/**
 * Slow DC tracker used to remove sensor offset without following respiration.
 *
 * The previous error-triggered fast mode was active during normal abdominal
 * breathing because its 60-ADC threshold was below the physiological signal
 * amplitude.  That made the reported `dc` value contain respiration.  A
 * bounded, fixed-rate update keeps the baseline slow and predictable.  The
 * first sample still initializes the tracker immediately.
 */
class DcBlocker(
    private val alpha: Float = 0.001f,
    private val maxStepPerSample: Float = 0.5f
) {
    var dc = 2600f
        private set
    private var firstSample = true

    fun next(x: Float): Float {
        // Initialize dc to first sample for instant convergence
        if (firstSample) {
            dc = x
            firstSample = false
        }
        val step = (alpha * (x - dc)).coerceIn(-maxStepPerSample, maxStepPerSample)
        dc += step
        return x - dc
    }

    fun clear() {
        dc = 2600f
        firstSample = true
    }
}
// Simple filters for separating respiration and heart signals.
class MovingAverage(private val windowSize: Int) {
    private val buf = FloatArray(windowSize)
    private var sum = 0f
    private var idx = 0
    private var filled = 0

    fun next(x: Float): Float {
        sum -= buf[idx]
        buf[idx] = x
        sum += x
        idx = (idx + 1) % windowSize
        if (filled < windowSize) filled++
        return sum / filled
    }
    fun clear() {
        buf.fill(0f)
        sum = 0f
        idx = 0
        filled = 0
    }
}

class ShortSmoother(windowSize: Int) {
    private val ma = MovingAverage(windowSize)
    fun next(x: Float) = ma.next(x)
    fun clear() = ma.clear()
}


class DualSmoother(windowSize: Int) {
    private val ma1 = MovingAverage(windowSize)
    private val ma2 = MovingAverage(windowSize)
    fun next(x: Float): Float = ma2.next(ma1.next(x))
    fun clear() {
        ma1.clear()
        ma2.clear()
    }
}

class HighPassFilter(private val alpha: Float) {
    private var yPrev = 0f
    private var xPrev = 0f

    fun next(x: Float): Float {
        // y[n] = alpha * (y[n-1] + x[n] - x[n-1])
        val y = alpha * (yPrev + x - xPrev)
        xPrev = x
        yPrev = y
        return y
    }

    fun clear() {
        yPrev = 0f
        xPrev = 0f
    }
}
class NotchFilter(private val freqHz: Float, private val sampleRate: Int, private val q: Float = 30f) {
    private val b = FloatArray(3); private val a = FloatArray(3)
    private var x1=0f; private var x2=0f; private var y1=0f; private var y2=0f
    init { val o=2.0*Math.PI*freqHz/sampleRate; val w=kotlin.math.sin(o)/(2f*q); val c=kotlin.math.cos(o); val d=1f+w.toFloat()
        b[0]=1f/d; b[1]=-2f*c.toFloat()/d; b[2]=1f/d; a[1]=-2f*c.toFloat()/d; a[2]=(1f-w.toFloat())/d }
    fun next(x:Float):Float{val y=b[0]*x+b[1]*x1+b[2]*x2-a[1]*y1-a[2]*y2;x2=x1;x1=x;y2=y1;y1=y;return y}
    fun clear(){x1=0f;x2=0f;y1=0f;y2=0f}
}
class LowPassFilter(private val cutoffHz: Float, private val sampleRate: Int) {
    private var yPrev=0f; private val alpha=(1f-kotlin.math.exp(-2.0*Math.PI*cutoffHz/sampleRate)).toFloat()
    fun next(x:Float):Float{val y=alpha*x+(1f-alpha)*yPrev;yPrev=y;return y}
    fun clear(){yPrev=0f}
}

/** Second-order Butterworth biquad used by the parallel heart-band experiment. */
class BiquadFilter private constructor(
    private val b0: Float,
    private val b1: Float,
    private val b2: Float,
    private val a1: Float,
    private val a2: Float
) {
    private var x1 = 0f
    private var x2 = 0f
    private var y1 = 0f
    private var y2 = 0f

    fun next(x: Float): Float {
        val y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        x2 = x1
        x1 = x
        y2 = y1
        y1 = y
        return y
    }

    fun clear() {
        x1 = 0f
        x2 = 0f
        y1 = 0f
        y2 = 0f
    }

    companion object {
        private const val BUTTERWORTH_Q = 0.70710678f

        fun lowPass(
            cutoffHz: Float,
            sampleRate: Int,
            q: Float = BUTTERWORTH_Q
        ): BiquadFilter {
            val omega = 2.0 * Math.PI * cutoffHz / sampleRate
            val cosOmega = kotlin.math.cos(omega).toFloat()
            val alpha = (kotlin.math.sin(omega) / (2.0 * q)).toFloat()
            val a0 = 1f + alpha
            return BiquadFilter(
                b0 = ((1f - cosOmega) / 2f) / a0,
                b1 = (1f - cosOmega) / a0,
                b2 = ((1f - cosOmega) / 2f) / a0,
                a1 = (-2f * cosOmega) / a0,
                a2 = (1f - alpha) / a0
            )
        }

        fun highPass(
            cutoffHz: Float,
            sampleRate: Int,
            q: Float = BUTTERWORTH_Q
        ): BiquadFilter {
            val omega = 2.0 * Math.PI * cutoffHz / sampleRate
            val cosOmega = kotlin.math.cos(omega).toFloat()
            val alpha = (kotlin.math.sin(omega) / (2.0 * q)).toFloat()
            val a0 = 1f + alpha
            return BiquadFilter(
                b0 = ((1f + cosOmega) / 2f) / a0,
                b1 = (-(1f + cosOmega)) / a0,
                b2 = ((1f + cosOmega) / 2f) / a0,
                a1 = (-2f * cosOmega) / a0,
                a2 = (1f - alpha) / a0
            )
        }
    }
}

class DifferentialHeartSeparator(private val windowSize: Int = 10, private val delaySteps: Int = 5) {
    // Stage 1: Moving Average (Ring Buffer + Running Sum, O(1))
    private val maBuf = FloatArray(windowSize)
    private var maIdx = 0
    private var maSum = 0f
    private var maCount = 0

    // Stage 2: raw-signal delay buffer (compensates the MA group delay)
    private val delayBuf = FloatArray(maxOf(1, delaySteps))
    private var delayIdx = 0
    private var delayCount = 0

    fun next(x: Float): Float {
        // Stage 1: 10-point Moving Average with running sum
        maSum -= maBuf[maIdx]
        maBuf[maIdx] = x
        maSum += x
        maIdx = (maIdx + 1) % windowSize
        if (maCount < windowSize) maCount++
        val mean = maSum / maCount

        // The paper delays the original signal, not the moving average:
        // d(n) = x(n-5) - MA10{x(n)} at 50 Hz.
        val delayedRaw: Float
        val delayReady: Boolean
        if (delaySteps == 0) {
            delayedRaw = x
            delayReady = true
        } else {
            delayedRaw = delayBuf[delayIdx]
            delayReady = delayCount >= delaySteps
            delayBuf[delayIdx] = x
            delayIdx = (delayIdx + 1) % delaySteps
            if (delayCount < delaySteps) delayCount++
        }

        return if (maCount >= windowSize && delayReady) delayedRaw - mean else 0f
    }

    fun clear() {
        maBuf.fill(0f); maIdx = 0; maSum = 0f; maCount = 0
        delayBuf.fill(0f); delayIdx = 0; delayCount = 0
    }
}
/** FIR 633??????
 *  Fpass = 0.4 Hz, Fstop = 0.6 Hz, Apass = 1 dB, Astop = 80 dB
 *  Fs = 50 Hz, Group delay = 633/(2*50) = 6.33 s
 *  Pre-computed Kaiser window coefficients
 */
val RESP_FIR_COEFFS = floatArrayOf(
  2.31374651e-06f,2.45283155e-06f,2.57378132e-06f,2.67302466e-06f,2.74687936e-06f,2.79157655e-06f,2.80328760e-06f,2.77815308e-06f,2.71231402e-06f,2.60194510e-06f,2.44328972e-06f,2.23269689e-06f,1.96665945e-06f,1.64185381e-06f,1.25518060e-06f,8.03806272e-07f,
  2.85205199e-07f,-3.02797945e-07f,-9.61985816e-07f,-1.69370639e-06f,-2.49883165e-06f,-3.37771726e-06f,-4.33016368e-06f,-5.35537899e-06f,-6.45194381e-06f,-7.61777877e-06f,-8.85011477e-06f,-1.01454665e-05f,-1.14996096e-05f,-1.29075618e-05f,-1.43635679e-05f,-1.58610903e-05f,
  -1.73928040e-05f,-1.89505961e-05f,-2.05255726e-05f,-2.21080693e-05f,-2.36876697e-05f,-2.52532287e-05f,-2.67929030e-05f,-2.82941879e-05f,-2.97439606e-05f,-3.11285302e-05f,-3.24336945e-05f,-3.36448033e-05f,-3.47468288e-05f,-3.57244412e-05f,-3.65620921e-05f,-3.72441028e-05f,
  -3.77547587e-05f,-3.80784095e-05f,-3.81995741e-05f,-3.81030497e-05f,-3.77740263e-05f,-3.71982037e-05f,-3.63619118e-05f,-3.52522339e-05f,-3.38571310e-05f,-3.21655678e-05f,-3.01676389e-05f,-2.78546942e-05f,-2.52194634e-05f,-2.22561784e-05f,-1.89606930e-05f,-1.53305977e-05f,
  -1.13653313e-05f,-7.06628606e-06f,-2.43690579e-06f,2.51722224e-06f,7.78828476e-06f,1.33661688e-05f,1.92383981e-05f,2.53900806e-05f,3.18038680e-05f,3.84599272e-05f,4.53359254e-05f,5.24070300e-05f,5.96459216e-05f,6.70228236e-05f,7.45055471e-05f,8.20595517e-05f,
  8.96480230e-05f,9.72319674e-05f,1.04770324e-04f,1.12220091e-04f,1.19536474e-04f,1.26673049e-04f,1.33581939e-04f,1.40214013e-04f,1.46519097e-04f,1.52446202e-04f,1.57943768e-04f,1.62959918e-04f,1.67442732e-04f,1.71340529e-04f,1.74602159e-04f,1.77177311e-04f,
  1.79016822e-04f,1.80073000e-04f,1.80299951e-04f,1.79653911e-04f,1.78093576e-04f,1.75580439e-04f,1.72079127e-04f,1.67557729e-04f,1.61988122e-04f,1.55346294e-04f,1.47612656e-04f,1.38772342e-04f,1.28815495e-04f,1.17737545e-04f,1.05539466e-04f,9.22280090e-05f,
  7.78159244e-05f,6.23221543e-05f,4.57720020e-05f,2.81972756e-05f,9.63640213e-06f,-9.86548792e-06f,-3.02565078e-05f,-5.14779952e-05f,-7.34645270e-05f,-9.61439705e-05f,-1.19437571e-04f,-1.43260076e-04f,-1.67519902e-04f,-1.92119336e-04f,-2.16954776e-04f,-2.41917017e-04f,
  -2.66891570e-04f,-2.91759026e-04f,-3.16395452e-04f,-3.40672833e-04f,-3.64459550e-04f,-3.87620886e-04f,-4.10019579e-04f,-4.31516400e-04f,-4.51970769e-04f,-4.71241392e-04f,-4.89186936e-04f,-5.05666721e-04f,-5.20541438e-04f,-5.33673887e-04f,-5.44929727e-04f,-5.54178249e-04f,
  -5.61293145e-04f,-5.66153298e-04f,-5.68643564e-04f,-5.68655557e-04f,-5.66088430e-04f,-5.60849645e-04f,-5.52855732e-04f,-5.42033026e-04f,-5.28318388e-04f,-5.11659897e-04f,-4.92017513e-04f,-4.69363705e-04f,-4.43684041e-04f,-4.14977735e-04f,-3.83258149e-04f,-3.48553239e-04f,
  -3.10905959e-04f,-2.70374588e-04f,-2.27033015e-04f,-1.80970944e-04f,-1.32294042e-04f,-8.11240091e-05f,-2.75985797e-05f,2.81285554e-05f,8.58879013e-05f,1.45494392e-04f,2.06747705e-04f,2.69432653e-04f,3.33319666e-04f,3.98165345e-04f,4.63713111e-04f,5.29693922e-04f,
  5.95827083e-04f,6.61821126e-04f,7.27374778e-04f,7.92177994e-04f,8.55913071e-04f,9.18255831e-04f,9.78876866e-04f,1.03744285e-03f,1.09361793e-03f,1.14706510e-03f,1.19744775e-03f,1.24443111e-03f,1.28768388e-03f,1.32687974e-03f,1.36169907e-03f,1.39183049e-03f,
  1.41697257e-03f,1.43683548e-03f,1.45114262e-03f,1.45963233e-03f,1.46205943e-03f,1.45819694e-03f,1.44783758e-03f,1.43079536e-03f,1.40690703e-03f,1.37603357e-03f,1.33806155e-03f,1.29290445e-03f,1.24050388e-03f,1.18083075e-03f,1.11388633e-03f,1.03970319e-03f,
  9.58346110e-04f,8.69912788e-04f,7.74534493e-04f,6.72376579e-04f,5.63638867e-04f,4.48555907e-04f,3.27397098e-04f,2.00466669e-04f,6.81035246e-05f,-6.93190639e-05f,-2.11393903e-04f,-3.57680478e-04f,-5.07705696e-04f,-6.60964782e-04f,-8.16922322e-04f,-9.75013453e-04f,
  -1.13464520e-03f,-1.29519794e-03f,-1.45602706e-03f,-1.61646465e-03f,-1.77582145e-03f,-1.93338884e-03f,-2.08844094e-03f,-2.24023694e-03f,-2.38802337e-03f,-2.53103662e-03f,-2.66850548e-03f,-2.79965375e-03f,-2.92370298e-03f,-3.03987524e-03f,-3.14739596e-03f,-3.24549680e-03f,
  -3.33341858e-03f,-3.41041419e-03f,-3.47575160e-03f,-3.52871677e-03f,-3.56861659e-03f,-3.59478186e-03f,-3.60657012e-03f,-3.60336855e-03f,-3.58459671e-03f,-3.54970931e-03f,-3.49819879e-03f,-3.42959796e-03f,-3.34348234e-03f,-3.23947258e-03f,-3.11723662e-03f,-2.97649176e-03f,
  -2.81700661e-03f,-2.63860287e-03f,-2.44115688e-03f,-2.22460112e-03f,-1.98892544e-03f,-1.73417811e-03f,-1.46046675e-03f,-1.16795893e-03f,-8.56882708e-04f,-5.27526859e-04f,-1.80240930e-04f,1.84564926e-04f,5.66420330e-04f,9.64795281e-04f,1.37910098e-03f,1.80869084e-03f,
  2.25286180e-03f,2.71085572e-03f,3.18186112e-03f,3.66501502e-03f,4.15940507e-03f,4.66407179e-03f,5.17801106e-03f,5.70017680e-03f,6.22948374e-03f,6.76481047e-03f,7.30500255e-03f,7.84887580e-03f,8.39521975e-03f,8.94280117e-03f,9.49036771e-03f,1.00366517e-02f,
  1.05803739e-02f,1.11202476e-02f,1.16549822e-02f,1.21832878e-02f,1.27038786e-02f,1.32154773e-02f,1.37168193e-02f,1.42066563e-02f,1.46837605e-02f,1.51469287e-02f,1.55949860e-02f,1.60267897e-02f,1.64412329e-02f,1.68372483e-02f,1.72138115e-02f,1.75699447e-02f,
  1.79047196e-02f,1.82172604e-02f,1.85067474e-02f,1.87724186e-02f,1.90135734e-02f,1.92295742e-02f,1.94198486e-02f,1.95838916e-02f,1.97212672e-02f,1.98316098e-02f,1.99146252e-02f,1.99700923e-02f,1.99978628e-02f,1.99978628e-02f,1.99700923e-02f,1.99146252e-02f,
  1.98316098e-02f,1.97212672e-02f,1.95838916e-02f,1.94198486e-02f,1.92295742e-02f,1.90135734e-02f,1.87724186e-02f,1.85067474e-02f,1.82172604e-02f,1.79047196e-02f,1.75699447e-02f,1.72138115e-02f,1.68372483e-02f,1.64412329e-02f,1.60267897e-02f,1.55949860e-02f,
  1.51469287e-02f,1.46837605e-02f,1.42066563e-02f,1.37168193e-02f,1.32154773e-02f,1.27038786e-02f,1.21832878e-02f,1.16549822e-02f,1.11202476e-02f,1.05803739e-02f,1.00366517e-02f,9.49036771e-03f,8.94280117e-03f,8.39521975e-03f,7.84887580e-03f,7.30500255e-03f,
  6.76481047e-03f,6.22948374e-03f,5.70017680e-03f,5.17801106e-03f,4.66407179e-03f,4.15940507e-03f,3.66501502e-03f,3.18186112e-03f,2.71085572e-03f,2.25286180e-03f,1.80869084e-03f,1.37910098e-03f,9.64795281e-04f,5.66420330e-04f,1.84564926e-04f,-1.80240930e-04f,
  -5.27526859e-04f,-8.56882708e-04f,-1.16795893e-03f,-1.46046675e-03f,-1.73417811e-03f,-1.98892544e-03f,-2.22460112e-03f,-2.44115688e-03f,-2.63860287e-03f,-2.81700661e-03f,-2.97649176e-03f,-3.11723662e-03f,-3.23947258e-03f,-3.34348234e-03f,-3.42959796e-03f,-3.49819879e-03f,
  -3.54970931e-03f,-3.58459671e-03f,-3.60336855e-03f,-3.60657012e-03f,-3.59478186e-03f,-3.56861659e-03f,-3.52871677e-03f,-3.47575160e-03f,-3.41041419e-03f,-3.33341858e-03f,-3.24549680e-03f,-3.14739596e-03f,-3.03987524e-03f,-2.92370298e-03f,-2.79965375e-03f,-2.66850548e-03f,
  -2.53103662e-03f,-2.38802337e-03f,-2.24023694e-03f,-2.08844094e-03f,-1.93338884e-03f,-1.77582145e-03f,-1.61646465e-03f,-1.45602706e-03f,-1.29519794e-03f,-1.13464520e-03f,-9.75013453e-04f,-8.16922322e-04f,-6.60964782e-04f,-5.07705696e-04f,-3.57680478e-04f,-2.11393903e-04f,
  -6.93190639e-05f,6.81035246e-05f,2.00466669e-04f,3.27397098e-04f,4.48555907e-04f,5.63638867e-04f,6.72376579e-04f,7.74534493e-04f,8.69912788e-04f,9.58346110e-04f,1.03970319e-03f,1.11388633e-03f,1.18083075e-03f,1.24050388e-03f,1.29290445e-03f,1.33806155e-03f,
  1.37603357e-03f,1.40690703e-03f,1.43079536e-03f,1.44783758e-03f,1.45819694e-03f,1.46205943e-03f,1.45963233e-03f,1.45114262e-03f,1.43683548e-03f,1.41697257e-03f,1.39183049e-03f,1.36169907e-03f,1.32687974e-03f,1.28768388e-03f,1.24443111e-03f,1.19744775e-03f,
  1.14706510e-03f,1.09361793e-03f,1.03744285e-03f,9.78876866e-04f,9.18255831e-04f,8.55913071e-04f,7.92177994e-04f,7.27374778e-04f,6.61821126e-04f,5.95827083e-04f,5.29693922e-04f,4.63713111e-04f,3.98165345e-04f,3.33319666e-04f,2.69432653e-04f,2.06747705e-04f,
  1.45494392e-04f,8.58879013e-05f,2.81285554e-05f,-2.75985797e-05f,-8.11240091e-05f,-1.32294042e-04f,-1.80970944e-04f,-2.27033015e-04f,-2.70374588e-04f,-3.10905959e-04f,-3.48553239e-04f,-3.83258149e-04f,-4.14977735e-04f,-4.43684041e-04f,-4.69363705e-04f,-4.92017513e-04f,
  -5.11659897e-04f,-5.28318388e-04f,-5.42033026e-04f,-5.52855732e-04f,-5.60849645e-04f,-5.66088430e-04f,-5.68655557e-04f,-5.68643564e-04f,-5.66153298e-04f,-5.61293145e-04f,-5.54178249e-04f,-5.44929727e-04f,-5.33673887e-04f,-5.20541438e-04f,-5.05666721e-04f,-4.89186936e-04f,
  -4.71241392e-04f,-4.51970769e-04f,-4.31516400e-04f,-4.10019579e-04f,-3.87620886e-04f,-3.64459550e-04f,-3.40672833e-04f,-3.16395452e-04f,-2.91759026e-04f,-2.66891570e-04f,-2.41917017e-04f,-2.16954776e-04f,-1.92119336e-04f,-1.67519902e-04f,-1.43260076e-04f,-1.19437571e-04f,
  -9.61439705e-05f,-7.34645270e-05f,-5.14779952e-05f,-3.02565078e-05f,-9.86548792e-06f,9.63640213e-06f,2.81972756e-05f,4.57720020e-05f,6.23221543e-05f,7.78159244e-05f,9.22280090e-05f,1.05539466e-04f,1.17737545e-04f,1.28815495e-04f,1.38772342e-04f,1.47612656e-04f,
  1.55346294e-04f,1.61988122e-04f,1.67557729e-04f,1.72079127e-04f,1.75580439e-04f,1.78093576e-04f,1.79653911e-04f,1.80299951e-04f,1.80073000e-04f,1.79016822e-04f,1.77177311e-04f,1.74602159e-04f,1.71340529e-04f,1.67442732e-04f,1.62959918e-04f,1.57943768e-04f,
  1.52446202e-04f,1.46519097e-04f,1.40214013e-04f,1.33581939e-04f,1.26673049e-04f,1.19536474e-04f,1.12220091e-04f,1.04770324e-04f,9.72319674e-05f,8.96480230e-05f,8.20595517e-05f,7.45055471e-05f,6.70228236e-05f,5.96459216e-05f,5.24070300e-05f,4.53359254e-05f,
  3.84599272e-05f,3.18038680e-05f,2.53900806e-05f,1.92383981e-05f,1.33661688e-05f,7.78828476e-06f,2.51722224e-06f,-2.43690579e-06f,-7.06628606e-06f,-1.13653313e-05f,-1.53305977e-05f,-1.89606930e-05f,-2.22561784e-05f,-2.52194634e-05f,-2.78546942e-05f,-3.01676389e-05f,
  -3.21655678e-05f,-3.38571310e-05f,-3.52522339e-05f,-3.63619118e-05f,-3.71982037e-05f,-3.77740263e-05f,-3.81030497e-05f,-3.81995741e-05f,-3.80784095e-05f,-3.77547587e-05f,-3.72441028e-05f,-3.65620921e-05f,-3.57244412e-05f,-3.47468288e-05f,-3.36448033e-05f,-3.24336945e-05f,
  -3.11285302e-05f,-2.97439606e-05f,-2.82941879e-05f,-2.67929030e-05f,-2.52532287e-05f,-2.36876697e-05f,-2.21080693e-05f,-2.05255726e-05f,-1.89505961e-05f,-1.73928040e-05f,-1.58610903e-05f,-1.43635679e-05f,-1.29075618e-05f,-1.14996096e-05f,-1.01454665e-05f,-8.85011477e-06f,
  -7.61777877e-06f,-6.45194381e-06f,-5.35537899e-06f,-4.33016368e-06f,-3.37771726e-06f,-2.49883165e-06f,-1.69370639e-06f,-9.61985816e-07f,-3.02797945e-07f,2.85205199e-07f,8.03806272e-07f,1.25518060e-06f,1.64185381e-06f,1.96665945e-06f,2.23269689e-06f,2.44328972e-06f,
  2.60194510e-06f,2.71231402e-06f,2.77815308e-06f,2.80328760e-06f,2.79157655e-06f,2.74687936e-06f,2.67302466e-06f,2.57378132e-06f,2.45283155e-06f,2.31374651e-06f,
)

class FIRFilter(private val coeffs: FloatArray) {
    private val n = coeffs.size
    private val delayLine = FloatArray(n)
    private var idx = 0
    private var filled = 0

    fun next(x: Float): Float {
        delayLine[idx] = x
        idx = (idx + 1) % n
        if (filled < n) filled++
        if (filled < n) return 0f

        var y = 0f
        var j = idx
        for (c in coeffs) {
            y += c * delayLine[j]
            j = (j + 1) % n
        }
        return y
    }

    fun clear() {
        delayLine.fill(0f); idx = 0; filled = 0
    }
}
class DualPeakDetector(
    private val thresholdRatio: Float = 0.15f,
    private val amplitudeWindowSamples: Int = 250
) {
    // --- First pass: basic local maximum detection ---
    private var x0 = 0f
    private var x1 = 0f
    private var trend = 0          // -1 falling, 0 initial, 1 rising
    private var peakT = 0L

    // Adaptive amplitude tracking (EMA of |hr signal|)
    private var hrAmpEma = 10f     // initial floor

    // Candidate peak buffer
    private val candidates = mutableListOf<Pair<Long, Float>>()

    // --- Second pass: envelope peak -> AO peak detection ---
    private var prevAOT = 0L

    // Recent valid BPM history for RR consistency check
    private val rrHistory = mutableListOf<Float>()

    // Physiological constraints
    private val minRR = 14          // ~300ms at 50Hz (180 BPM upper bound)
    private val maxRR = 75          // ~1500ms at 50Hz (40 BPM lower bound)

    private fun adaptiveThreshold(): Float {
        return maxOf(hrAmpEma * thresholdRatio, 5f)
    }

    fun next(x: Float, t: Long): Float? {
        x1 = x0
        x0 = x
        val df = x0 - x1

        // Update adaptive amplitude estimate
        hrAmpEma += 0.05f * (kotlin.math.abs(x) - hrAmpEma)

        var hr: Float? = null

        // === First pass: find local maxima using adaptive threshold ===
        val thr = adaptiveThreshold()
        if (df > 0 && trend == -1) {
            trend = 1
        } else if (df < 0 && trend == 1) {
            if (x1 > thr && peakT > 0) {
                candidates.add(peakT to x1)
            }
            trend = -1
        } else if (trend == 0) {
            trend = if (df > 0) 1 else -1
        }
        if (trend == 1) peakT = t

        // === Second pass: AO peak from candidate envelope ===
        while (candidates.size >= 3) {
            val (_, a1) = candidates[candidates.size - 3]
            val (t2, a2) = candidates[candidates.size - 2]
            val (_, a3) = candidates[candidates.size - 1]

            if (a2 > a1 && a2 > a3) {
                if (prevAOT > 0) {
                    val deltaN = ((t2 - prevAOT) / 20f + 0.5f).toInt()
                    if (deltaN in minRR..maxRR) {
                        val bpm = 3000f / deltaN
                        if (bpm in 40f..180f) {
                            // RR interval consistency check
                            if (rrHistory.size >= 3) {
                                val medianRR = rrHistory.sorted()[rrHistory.size / 2]
                                val deviation = kotlin.math.abs(bpm - medianRR)
                                val maxDeviation = medianRR * 0.20f
                                if (deviation <= maxDeviation) {
                                    hr = bpm
                                    rrHistory.add(bpm)
                                    if (rrHistory.size > 5) rrHistory.removeAt(0)
                                }
                            } else {
                                hr = bpm
                                rrHistory.add(bpm)
                            }
                        }
                    }
                }
                prevAOT = t2
                candidates.removeAll { it.first <= t2 }
            } else {
                break
            }
        }

        if (candidates.size > 50) candidates.clear()
        return hr
    }

    fun clear() {
        x0 = 0f; x1 = 0f; trend = 0; peakT = 0L
        candidates.clear(); prevAOT = 0L
        rrHistory.clear()
        hrAmpEma = 10f
    }
}

data class WindowedHeartDetection(
    val detected: Boolean = false,
    val bpm: Float? = null,
    val peakTimeMs: Long? = null,
    /** Interval regularity and history support in [0, 1]. */
    val quality: Float = 0f,
    val intervalCount: Int = 0
)

/**
 * Parallel diagnostic detector for the normalized display waveform.
 * Local maxima arriving within one mechanical heart complex are grouped first;
 * only the strongest maximum is accepted after the group window expires.
 */
class WindowedHeartPeakDetector(
    private val threshold: Float = 0.3f,
    private val groupWindowMs: Long = 300L,
    private val refractoryMs: Long = 500L,
    private val minIntervalMs: Long = refractoryMs,
    private val maxIntervalMs: Long = 1_500L
) {
    private var previous2 = 0f
    private var previous1 = 0f
    private var previous1TimeMs = 0L
    private var sampleCount = 0

    private var pendingTimeMs = 0L
    private var pendingAmplitude = Float.NEGATIVE_INFINITY
    private var lastAcceptedTimeMs = 0L
    private val recentIntervalsMs = ArrayDeque<Long>()

    fun next(x: Float, timeMs: Long): WindowedHeartDetection {
        var result = WindowedHeartDetection()

        if (sampleCount >= 2 && previous1 > previous2 && previous1 >= x && previous1 > threshold) {
            if (pendingTimeMs == 0L) {
                pendingTimeMs = previous1TimeMs
                pendingAmplitude = previous1
            } else if (previous1TimeMs - pendingTimeMs <= groupWindowMs) {
                if (previous1 > pendingAmplitude) {
                    pendingTimeMs = previous1TimeMs
                    pendingAmplitude = previous1
                }
            } else {
                result = acceptPending()
                pendingTimeMs = previous1TimeMs
                pendingAmplitude = previous1
            }
        }

        if (pendingTimeMs > 0L && timeMs - pendingTimeMs >= groupWindowMs) {
            result = acceptPending()
        }

        previous2 = previous1
        previous1 = x
        previous1TimeMs = timeMs
        sampleCount++
        return result
    }

    private fun acceptPending(): WindowedHeartDetection {
        val peakTimeMs = pendingTimeMs
        pendingTimeMs = 0L
        pendingAmplitude = Float.NEGATIVE_INFINITY
        if (peakTimeMs <= 0L ||
            (lastAcceptedTimeMs > 0L && peakTimeMs - lastAcceptedTimeMs < refractoryMs)
        ) {
            return WindowedHeartDetection()
        }

        var bpm: Float? = null
        var quality = 0f
        if (lastAcceptedTimeMs > 0L) {
            val intervalMs = peakTimeMs - lastAcceptedTimeMs
            if (intervalMs in minIntervalMs..maxIntervalMs) {
                recentIntervalsMs.addLast(intervalMs)
                while (recentIntervalsMs.size > 5) recentIntervalsMs.removeFirst()
                val sorted = recentIntervalsMs.sorted()
                val medianMs = sorted[sorted.size / 2]
                bpm = 60_000f / medianMs
                val mean = recentIntervalsMs.average()
                val variance = recentIntervalsMs.sumOf {
                    val difference = it - mean
                    difference * difference
                } / recentIntervalsMs.size
                val coefficientOfVariation = if (mean > 0.0) {
                    kotlin.math.sqrt(variance) / mean
                } else 1.0
                val historySupport = (recentIntervalsMs.size / 4f).coerceIn(0f, 1f)
                val regularity = (1.0 - coefficientOfVariation / 0.25)
                    .coerceIn(0.0, 1.0).toFloat()
                quality = historySupport * regularity
            }
        }
        lastAcceptedTimeMs = peakTimeMs
        return WindowedHeartDetection(
            detected = true,
            bpm = bpm,
            peakTimeMs = peakTimeMs,
            quality = quality,
            intervalCount = recentIntervalsMs.size
        )
    }

    fun clear() {
        previous2 = 0f
        previous1 = 0f
        previous1TimeMs = 0L
        sampleCount = 0
        pendingTimeMs = 0L
        pendingAmplitude = Float.NEGATIVE_INFINITY
        lastAcceptedTimeMs = 0L
        recentIntervalsMs.clear()
    }
}

class BreathingRateDetector {
    private var data0 = 0f
    private var data1 = 0f
    private var state = 0         // 1=increasing, -1=decreasing, 0=initial
    private var prevExtremumT = 0L  // t0: last extremum timestamp
    private var sampleCount = 0

    // Half-period buffer (in ms)
    private val halfPeriods = mutableListOf<Float>()
    private val maxHalfPeriods = 40

    // Amplitude tracking: extreme values use data1 (last sample before turn)
    private var prevPeakAmp = 0f
    private var prevValleyAmp = 0f
    private var respP2p = 100f

    fun next(x: Float, t: Long): Float? {
        data1 = data0
        data0 = x
        sampleCount++
        // Update respiratory P2P estimate for amplitude gating
        if (sampleCount % 50 == 0) {
            // Simplified: use approximate P2P from recent samples
            respP2p += 0.1f * (kotlin.math.abs(data0 - data1) * 10f - respP2p)
        }

        // First 2 samples: establish initial trend, no detection
        if (sampleCount <= 2) {
            if (sampleCount == 2) {
                state = if (data0 >= data1) 1 else -1
            }
            return null
        }

        val diff = data0 - data1
        var rpm: Float? = null

        // Detect trend reversals via first-difference sign change
        if (diff > 0 && state == -1) {
            // Valley: trend changed from decreasing to increasing
            // data1 is the valley amplitude (last sample before the turn)
            prevValleyAmp = data1
            // Amplitude gating: only count if this valley is significant vs previous peak
            val valleyAmpOk = prevPeakAmp == 0f || kotlin.math.abs(data1 - prevPeakAmp) > respP2p * 0.20f
            if (valleyAmpOk) {
                if (prevExtremumT > 0) {
                    val halfMs = t - prevExtremumT
                    if (halfMs >= 1000L) {
                        halfPeriods.add(halfMs.toFloat())
                        if (halfPeriods.size > maxHalfPeriods) halfPeriods.removeAt(0)
                    }
                }
                prevExtremumT = t
                state = 1
            }
        } else if (diff < 0 && state == 1) {
            // Peak: trend changed from increasing to decreasing
            // data1 is the peak amplitude (last sample before the turn)
            prevPeakAmp = data1
            val peakAmpOk = prevValleyAmp == 0f || kotlin.math.abs(data1 - prevValleyAmp) > respP2p * 0.20f
            if (peakAmpOk) {
                if (prevExtremumT > 0) {
                    val halfMs = t - prevExtremumT
                    if (halfMs >= 1000L) {
                        halfPeriods.add(halfMs.toFloat())
                        if (halfPeriods.size > maxHalfPeriods) halfPeriods.removeAt(0)
                    }
                }
                prevExtremumT = t
                state = -1
            }
        } else if (diff > 0 && state == 0) {
            state = 1
        } else if (diff < 0 && state == 0) {
            state = -1
        }

        // RPM = 30000 / avg_half_period_ms, using trimmed mean:
        // remove first 3 and last 3 half-periods (by sorted value), average the rest
        val trimCount = (halfPeriods.size * 0.15f).toInt().coerceIn(1, 5)
        if (halfPeriods.size >= trimCount * 2 + 2) {
            val sorted = halfPeriods.sorted()
            val trimmed = sorted.drop(trimCount).dropLast(trimCount)
            if (trimmed.isNotEmpty()) {
                val avgHalfMs = trimmed.sum() / trimmed.size
                if (avgHalfMs > 0f) {
                    val rawRpm = 30000f / avgHalfMs
                    if (rawRpm in 6f..30f) {
                        rpm = rawRpm
                    }
                }
            }
        }

        return rpm
    }

    fun clear() {
        data0 = 0f; data1 = 0f; state = 0
        prevExtremumT = 0L; sampleCount = 0
        halfPeriods.clear()
        prevPeakAmp = 0f; prevValleyAmp = 0f; respP2p = 100f
    }
}

/** Zero-crossing based RPM estimator.
 *  Counts positive zero crossings of the respiratory signal over a
 *  sliding window, then converts to RPM: RPM = count * 60 / windowSec.
 *  Each positive-going zero crossing represents one respiratory cycle.
 *  Immune to intra-cycle sub-peaks that inflate derivative-based detectors.
 */
class ZeroCrossingRpm(
    private val windowSec: Int = 30,
    private val fsHz: Int = 50
) {
    private val crossings = ArrayDeque<Long>()
    private var prevResp = 0f

    fun next(resp: Float, tMs: Long): Float? {
        // Detect positive-going zero crossing
        if (prevResp <= 0f && resp > 0f) {
            crossings.addLast(tMs)
        }
        prevResp = resp

        // Remove crossings outside the window
        val cutoff = tMs - windowSec * 1000L
        while (crossings.isNotEmpty() && crossings.first() < cutoff) {
            crossings.removeFirst()
        }

        // RPM = positive crossings * (60 / windowSec)
        return if (crossings.size >= 2) {
            crossings.size.toFloat() * 60f / windowSec
        } else null
    }

    fun clear() {
        crossings.clear()
        prevResp = 0f
    }
}

data class RespCycleDetection(
    val crossing: Boolean = false,
    val rpm: Float? = null
)

/**
 * Low-delay startup estimator for respiration.
 *
 * A first value is emitted only after peak-valley-peak (or the inverse), so it
 * still observes one complete breath. Two adjacent half periods must agree;
 * the full-cycle Schmitt detector remains the preferred long-term estimator.
 */
class RespirationStartupRateDetector(
    private val minHalfCycleMs: Long = 1_000L,
    private val maxHalfCycleMs: Long = 5_000L,
    private val maximumHalfCycleRatio: Float = 1.35f,
    private val minimumSwing: Float = 10f
) {
    private var previousValue = 0f
    private var previousTimeMs = 0L
    private var trend = 0
    private var sampleCount = 0
    private var lastExtremumTimeMs = 0L
    private var lastExtremumValue = 0f
    private var previousHalfCycleMs: Long? = null

    fun next(value: Float, timeMs: Long): Float? {
        if (sampleCount == 0) {
            previousValue = value
            previousTimeMs = timeMs
            sampleCount = 1
            return null
        }

        val difference = value - previousValue
        val nextTrend = when {
            difference > 1e-5f -> 1
            difference < -1e-5f -> -1
            else -> trend
        }
        var rpm: Float? = null
        if (trend != 0 && nextTrend != 0 && nextTrend != trend) {
            val extremumTimeMs = previousTimeMs
            val extremumValue = previousValue
            if (lastExtremumTimeMs == 0L) {
                lastExtremumTimeMs = extremumTimeMs
                lastExtremumValue = extremumValue
            } else {
                val halfCycleMs = extremumTimeMs - lastExtremumTimeMs
                val swing = kotlin.math.abs(extremumValue - lastExtremumValue)
                if (halfCycleMs in minHalfCycleMs..maxHalfCycleMs &&
                    swing >= minimumSwing
                ) {
                    val previousHalf = previousHalfCycleMs
                    if (previousHalf != null) {
                        val ratio = maxOf(previousHalf, halfCycleMs).toFloat() /
                            minOf(previousHalf, halfCycleMs).toFloat()
                        if (ratio <= maximumHalfCycleRatio) {
                            rpm = (30_000f / halfCycleMs).takeIf { it in 6f..30f }
                        }
                    }
                    previousHalfCycleMs = halfCycleMs
                    lastExtremumTimeMs = extremumTimeMs
                    lastExtremumValue = extremumValue
                } else if (halfCycleMs > maxHalfCycleMs * 2L) {
                    previousHalfCycleMs = null
                    lastExtremumTimeMs = extremumTimeMs
                    lastExtremumValue = extremumValue
                }
            }
        }
        trend = nextTrend
        previousValue = value
        previousTimeMs = timeMs
        sampleCount++
        return rpm
    }

    fun clear() {
        previousValue = 0f
        previousTimeMs = 0L
        trend = 0
        sampleCount = 0
        lastExtremumTimeMs = 0L
        lastExtremumValue = 0f
        previousHalfCycleMs = null
    }
}

/**
 * Full-cycle respiratory-rate estimator. A slowly tracked center and adaptive
 * Schmitt hysteresis suppress small zero-crossing chatter. One armed
 * negative-to-positive crossing represents one complete breath.
 */
class RespCycleRateDetector(
    private val minCycleMs: Long = 2_000L,
    private val maxCycleMs: Long = 10_000L,
    private val historySize: Int = 5,
    private val centerAlpha: Float = 0.0005f,
    private val envelopeAlpha: Float = 0.01f,
    private val hysteresisFraction: Float = 0.20f,
    private val minHysteresis: Float = 2f
) {
    private var center = 0f
    private var envelope = 0f
    private var initialized = false
    private var armedBelow = false
    private var lastCrossingTimeMs = 0L
    private val cycleHistoryMs = ArrayDeque<Long>()

    fun next(resp: Float, timeMs: Long): RespCycleDetection {
        if (!initialized) {
            center = resp
            initialized = true
            return RespCycleDetection()
        }

        center += centerAlpha * (resp - center)
        val centered = resp - center
        envelope += envelopeAlpha * (kotlin.math.abs(centered) - envelope)
        val hysteresis = maxOf(minHysteresis, envelope * hysteresisFraction)

        if (centered <= -hysteresis) armedBelow = true
        val crossed = armedBelow && centered >= hysteresis
        if (!crossed) return RespCycleDetection()
        armedBelow = false

        var rpm: Float? = null
        if (lastCrossingTimeMs > 0L) {
            val cycleMs = timeMs - lastCrossingTimeMs
            if (cycleMs in minCycleMs..maxCycleMs) {
                cycleHistoryMs.addLast(cycleMs)
                while (cycleHistoryMs.size > historySize) cycleHistoryMs.removeFirst()
                val sorted = cycleHistoryMs.sorted()
                val medianCycleMs = sorted[sorted.size / 2]
                rpm = 60_000f / medianCycleMs
            }
        }
        lastCrossingTimeMs = timeMs
        return RespCycleDetection(crossing = true, rpm = rpm)
    }

    fun clear() {
        center = 0f
        envelope = 0f
        initialized = false
        armedBelow = false
        lastCrossingTimeMs = 0L
        cycleHistoryMs.clear()
    }
}

data class HeartPeriodicityResult(
    val bpm: Float? = null,
    val quality: Float? = null,
    val candidates: List<HeartRateCandidateEvidence> = emptyList()
)

/**
 * Heart-rate estimator based on waveform periodicity rather than individual
 * threshold crossings. A longer window is used because a four-second window
 * contains too few slow heart cycles to resolve half/double-rate ambiguity.
 */
class HeartPeriodicityEstimator(
    private val fsHz: Int,
    private val windowSeconds: Int = 15,
    private val updateIntervalMs: Long = 1_000L,
    private val minBpm: Float = 40f,
    private val maxBpm: Float = 110f,
    private val minimumQuality: Float = 0.18f
) {
    private val samples = ArrayDeque<Float>()
    private val capacity = fsHz * windowSeconds
    private var lastSampleTimeMs = 0L
    private var lastUpdateTimeMs = 0L
    private var lastResult = HeartPeriodicityResult()

    fun next(value: Float, timeMs: Long): HeartPeriodicityResult {
        // Do not correlate samples from opposite sides of a BLE/motion gap.
        val expectedStepMs = 1_000L / fsHz
        if (lastSampleTimeMs > 0L && timeMs - lastSampleTimeMs > expectedStepMs * 3L) {
            clear()
        }
        lastSampleTimeMs = timeMs

        samples.addLast(value)
        while (samples.size > capacity) samples.removeFirst()
        if (samples.size < capacity || timeMs - lastUpdateTimeMs < updateIntervalMs) {
            return lastResult
        }
        lastUpdateTimeMs = timeMs

        val values = samples.toFloatArray()
        var mean = 0f
        for (sample in values) mean += sample
        mean /= values.size
        for (i in values.indices) values[i] -= mean

        val minLag = kotlin.math.ceil(fsHz * 60f / maxBpm).toInt()
        val maxLag = kotlin.math.floor(fsHz * 60f / minBpm).toInt()
        val correlations = FloatArray(maxLag + 1)
        var bestLag = 0
        var bestCorrelation = Float.NEGATIVE_INFINITY
        for (lag in minLag..maxLag) {
            var cross = 0.0
            var energyA = 0.0
            var energyB = 0.0
            for (i in lag until values.size) {
                val a = values[i].toDouble()
                val b = values[i - lag].toDouble()
                cross += a * b
                energyA += a * a
                energyB += b * b
            }
            val denominator = kotlin.math.sqrt(energyA * energyB)
            val correlation = if (denominator > 1e-9) (cross / denominator).toFloat() else 0f
            correlations[lag] = correlation
            if (correlation > bestCorrelation) {
                bestCorrelation = correlation
                bestLag = lag
            }
        }

        // A multi-feature BCG beat can make the two-beat lag slightly stronger
        // than the one-beat lag (for example, 80 bpm appearing as 40 bpm).
        // Prefer the half-lag only when its correlation is nearly as strong;
        // the formal BPM path still requires agreement with an independent
        // spectrum estimate.
        var selectedLag = bestLag
        var selectedCorrelation = bestCorrelation
        if (bestLag > 0) {
            val halfLag = bestLag / 2
            val from = maxOf(minLag, halfLag - 2)
            val to = minOf(maxLag, halfLag + 2)
            var harmonicLag = 0
            var harmonicCorrelation = Float.NEGATIVE_INFINITY
            for (candidateLag in from..to) {
                val candidateCorrelation = correlations[candidateLag]
                if (kotlin.math.abs(bestLag - candidateLag * 2) <= 3 &&
                    candidateCorrelation >= minimumQuality &&
                    candidateCorrelation >= bestCorrelation * 0.80f &&
                    candidateCorrelation > harmonicCorrelation
                ) {
                    harmonicLag = candidateLag
                    harmonicCorrelation = candidateCorrelation
                }
            }
            if (harmonicLag > 0) {
                selectedLag = harmonicLag
                selectedCorrelation = harmonicCorrelation
            }
        }

        val bpm = if (selectedLag > 0 && selectedCorrelation >= minimumQuality) {
            fsHz * 60f / selectedLag
        } else {
            null
        }
        val candidates = ArrayList<HeartRateCandidateEvidence>()
        for (lag in minLag..maxLag) {
            val correlation = correlations[lag]
            val left = if (lag > minLag) correlations[lag - 1] else Float.NEGATIVE_INFINITY
            val right = if (lag < maxLag) correlations[lag + 1] else Float.NEGATIVE_INFINITY
            if (correlation >= minimumQuality * 0.60f &&
                correlation >= left && correlation >= right
            ) {
                candidates += HeartRateCandidateEvidence(
                    bpm = fsHz * 60f / lag,
                    strength = correlation.coerceIn(0f, 1f)
                )
            }
        }
        val distinctCandidates = candidates
            .sortedByDescending { it.strength }
            .fold(ArrayList<HeartRateCandidateEvidence>()) { selected, candidate ->
                if (selected.none { kotlin.math.abs(it.bpm - candidate.bpm) < 4f }) {
                    selected += candidate
                }
                selected
            }
            .take(6)

        lastResult = HeartPeriodicityResult(
            bpm = bpm,
            quality = selectedCorrelation.coerceAtLeast(0f),
            candidates = distinctCandidates
        )
        return lastResult
    }

    fun clear() {
        samples.clear()
        lastSampleTimeMs = 0L
        lastUpdateTimeMs = 0L
        lastResult = HeartPeriodicityResult()
    }
}

data class HeartSpectrumResult(
    val bpm: Float? = null,
    /** Dominant-bin power divided by the median power in the search band. */
    val peakRatio: Float? = null,
    val candidates: List<HeartRateCandidateEvidence> = emptyList()
)

/**
 * Independent short-time spectrum estimator used to confirm autocorrelation.
 * It deliberately does not notch respiration harmonics: a harmonic can occupy
 * exactly the same frequency as the true heart fundamental and cannot then be
 * separated by a fixed frequency-domain hole.
 */
class HeartSpectrumEstimator(
    private val fsHz: Int,
    private val windowSeconds: Int = 10,
    private val updateIntervalMs: Long = 1_000L,
    private val minBpm: Float = 40f,
    private val maxBpm: Float = 110f,
    private val minimumPeakRatio: Float = 3f
) {
    private val samples = ArrayDeque<Float>()
    private val capacity = fsHz * windowSeconds
    private var lastSampleTimeMs = 0L
    private var lastUpdateTimeMs = 0L
    private var lastResult = HeartSpectrumResult()

    fun next(value: Float, timeMs: Long): HeartSpectrumResult {
        val expectedStepMs = 1_000L / fsHz
        if (lastSampleTimeMs > 0L && timeMs - lastSampleTimeMs > expectedStepMs * 3L) {
            clear()
        }
        lastSampleTimeMs = timeMs

        samples.addLast(value)
        while (samples.size > capacity) samples.removeFirst()
        if (samples.size < capacity || timeMs - lastUpdateTimeMs < updateIntervalMs) {
            return lastResult
        }
        lastUpdateTimeMs = timeMs

        val values = samples.toFloatArray()
        var mean = 0.0
        for (sample in values) mean += sample
        mean /= values.size

        // A fine frequency grid improves the displayed estimate without
        // pretending that the physical resolution exceeds the 10 s window.
        val minHz = minBpm / 60.0
        val maxHz = maxBpm / 60.0
        val stepHz = 0.01
        val binCount = kotlin.math.floor((maxHz - minHz) / stepHz).toInt() + 1
        val powers = FloatArray(binCount)
        var bestIndex = 0
        var bestPower = Float.NEGATIVE_INFINITY
        for (bin in 0 until binCount) {
            val frequency = minHz + bin * stepHz
            var real = 0.0
            var imaginary = 0.0
            for (index in values.indices) {
                val window = if (values.size > 1) {
                    0.5 - 0.5 * kotlin.math.cos(2.0 * Math.PI * index / (values.size - 1))
                } else 1.0
                val centered = (values[index] - mean) * window
                val angle = 2.0 * Math.PI * frequency * index / fsHz
                real += centered * kotlin.math.cos(angle)
                imaginary -= centered * kotlin.math.sin(angle)
            }
            val power = (real * real + imaginary * imaginary).toFloat()
            powers[bin] = power
            if (power > bestPower) {
                bestPower = power
                bestIndex = bin
            }
        }

        val sortedPowers = powers.copyOf().apply { sort() }
        val medianPower = if (sortedPowers.isNotEmpty()) {
            sortedPowers[sortedPowers.size / 2]
        } else 0f
        val peakRatio = if (medianPower > 1e-12f) bestPower / medianPower else 0f
        val bpm = if (bestPower > 1e-12f && peakRatio >= minimumPeakRatio) {
            ((minHz + bestIndex * stepHz) * 60.0).toFloat()
        } else null
        val candidates = ArrayList<HeartRateCandidateEvidence>()
        for (bin in powers.indices) {
            val power = powers[bin]
            val left = if (bin > 0) powers[bin - 1] else Float.NEGATIVE_INFINITY
            val right = if (bin < powers.lastIndex) powers[bin + 1] else Float.NEGATIVE_INFINITY
            val ratio = if (medianPower > 1e-12f) power / medianPower else 0f
            if (ratio >= 1.5f && power >= left && power >= right) {
                candidates += HeartRateCandidateEvidence(
                    bpm = ((minHz + bin * stepHz) * 60.0).toFloat(),
                    strength = ratio
                )
            }
        }
        val distinctCandidates = candidates
            .sortedByDescending { it.strength }
            .fold(ArrayList<HeartRateCandidateEvidence>()) { selected, candidate ->
                if (selected.none { kotlin.math.abs(it.bpm - candidate.bpm) < 4f }) {
                    selected += candidate
                }
                selected
            }
            .take(6)
        lastResult = HeartSpectrumResult(
            bpm = bpm,
            peakRatio = peakRatio,
            candidates = distinctCandidates
        )
        return lastResult
    }

    fun clear() {
        samples.clear()
        lastSampleTimeMs = 0L
        lastUpdateTimeMs = 0L
        lastResult = HeartSpectrumResult()
    }
}

