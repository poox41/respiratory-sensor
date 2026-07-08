package com.example.breathheartdemo
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
}


class DualSmoother(windowSize: Int) {
    private val ma1 = MovingAverage(windowSize)
    private val ma2 = MovingAverage(windowSize)
    fun next(x: Float): Float = ma2.next(ma1.next(x))
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
class DifferentialHeartSeparator(private val windowSize: Int = 10) {
    private val buf=FloatArray(windowSize); private var idx=0
    fun next(x:Float):Float{
        buf[idx]=x;idx=(idx+1)%windowSize;var s=0f;for(i in 0 until windowSize)s+=buf[i]
        val xBar=s/windowSize;s=0f;val st=(idx-5+windowSize)%windowSize
        for(i in 0 until windowSize)s+=buf[(st+i)%windowSize]
        return x-xBar+s/windowSize
    }
    fun clear(){buf.fill(0f);idx=0}
}
class FIRFilter(private val order: Int, private val cutoffHz: Float, private val sampleRate: Int) {
    private val coeffs=FloatArray(order+1); private val delayLine=FloatArray(order+1); private var idx=0; private var filled=0
    init{val o=2.0*Math.PI*cutoffHz/sampleRate;val m=order/2.0;for(idx in 0..order){val md=idx-m
        coeffs[idx]=if(kotlin.math.abs(md)<1e-6)(o/Math.PI).toFloat() else(kotlin.math.sin(o*md)/(Math.PI*md)).toFloat()
        coeffs[idx]*=(0.54-0.46*kotlin.math.cos(2.0*Math.PI*idx/order)).toFloat()}
        val s=coeffs.sum().let{if(it!=0f)it else 1f};for(i in coeffs.indices)coeffs[i]/=s}
    fun next(x:Float):Float{
        delayLine[idx]=x;idx=(idx+1)%delayLine.size;if(filled<delayLine.size)filled++;if(filled<delayLine.size)return 0f
        var y=0f;var j=idx;for(i in coeffs.indices){y+=coeffs[i]*delayLine[j];j=(j+1)%delayLine.size};return y}
    fun clear(){delayLine.fill(0f);idx=0;filled=0}
}
class DualPeakDetector(private val threshold: Float) {
    private var d0=0f;private var d1=0f;private var st=0;private var pkT=0L
    private val cand=mutableListOf<Pair<Long,Float>>();private var prevAO=0L
    fun next(x:Float,t:Long):Float?{
        d0=x;val df=d0-d1;var hr:Float?=null
        if(df>0&&st==-1){st=1}else if(df<0&&st==1){
            if(d1>threshold&&pkT>0)cand.add(pkT to d1);st=-1
        }else if(st==0)st=if(df>0)1 else -1
        if(st==1)pkT=t;d1=d0
        if(cand.size>=3){val(_,v1)=cand[cand.size-3];val(t2,v2)=cand[cand.size-2];val(t3,v3)=cand.last()
            if(v2>v1&&v2>v3){if(prevAO>0){val rr=t2-prevAO;if(rr>200L){val bpm=60000f/rr;if(bpm in 40f..180f)hr=bpm}}
            prevAO=t2;cand.removeAll{it.first<=t2}}}
        if(cand.size>50)cand.clear();return hr
    }
    fun clear(){d0=0f;d1=0f;st=0;pkT=0L;cand.clear();prevAO=0L}
}
class BreathingRateDetector {
    private var d0=0f;private var d1=0f;private var st=0
    private var pkT=0L;private var vlT=0L;private var prevPK=0L
    private var pkV=0f;private var vlV=0f;private var hc=0;private var sc=0
    fun next(x:Float,t:Long):Float?{
        d0=x;val df=d0-d1;var rpm:Float?=null
        if(df>0&&st==-1){vlT=t;vlV=x;hc++;st=1}else if(df<0&&st==1){pkV=x;hc++
            if(prevPK>0&&hc>=4){val pd=t-pkT;if(pd>0&&++sc>=3){val r=60000f/pd;if(r in 6f..30f)rpm=r}}
            prevPK=pkT;pkT=t;st=-1}else if(st==0)st=if(df>0)1 else -1
        d1=d0;return rpm
    }
    fun clear(){d0=0f;d1=0f;st=0;pkT=0L;vlT=0L;prevPK=0L;pkV=0f;vlV=0f;hc=0;sc=0}
}