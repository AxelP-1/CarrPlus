import math
import sympy
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt,fftconvolve
import random
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Audio

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band', output='sos')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5, padlen=33):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sosfiltfilt(sos, data, padlen=padlen)
    return y
  
def waveFolder(vals,foldfactor=1,mult=4,speedOfFolding=2,sinAmplitude=0.2):
  return [math.tanh(x*mult)+foldfactor**(-abs(x*mult))*math.sin(x*mult*speedOfFolding)*sinAmplitude for x in vals]

class Karplus:
  def __init__(self,bufferSize,attack=0.1,sampleRate=44100,release=0.1):
    self.buffer=[0]*bufferSize*sampleRate
    self.attack=attack
    self.sr=sampleRate
    self.t=0
    self.r=release

  def delay(self,samples,freq,decayPerSecond,hiCut):
    decay=(1/decayPerSecond)**(1/freq)
    realTime=int((1/freq)*self.sr)
    output=[]
    for i in range(len(samples)):
      self.buffer[self.t]=self.buffer[self.t-1]+math.tanh((samples[i]*(1-decay)+self.buffer[self.t-realTime]*decay-self.buffer[self.t-1])*hiCut/self.sr)*self.sr/hiCut
      output.append(self.buffer[self.t-realTime])
      self.t+=1
      self.t=self.t%len(self.buffer)
    return output
  def shortNoiseBurst(self):
    return [random.random()*(self.attack-i/self.sr) for i in range(int(self.attack*self.sr))]
  def KarplusGenerate(self,Hz,decay=2,length=2,hiCut=100000000000):
    data=self.shortNoiseBurst()
    length*=self.sr
    while len(data)<length:
      data.append(0)
    data=self.delay(data,Hz,decay,hiCut)
    for i in range(min(int(self.r*self.sr),len(data))):
      data[-1-i]*=i/min(int(self.r*self.sr),len(data))
    self.buffer=[0]*len(self.buffer)
    return data
  
def rhythmicVarispeed(samples,speeds,times,startTime=0,sr=44100):
  if not 0 in times:
    times=[0]+times
    speeds=[1]+speeds
  time=0
  speed=1
  startTimeInSample=int(sr*startTime)
  pointer=0
  output=[]
  while not time>times[-1]:
    while pointer>=len(samples):
      pointer-=len(samples)
    while pointer<0:
      pointer+=len(samples)
    output.append(samples[int(pointer)]*(1-pointer+int(pointer))+samples[math.ceil(pointer)%len(samples)]*(pointer-int(pointer)))
    loc=closest(times,time)
    if times[loc]==time:
      speed=speeds[loc]
    elif times[loc]>time:
      gap=times[loc]-times[loc-1]
      speed=speeds[loc]*(time-times[loc-1])/gap+speeds[loc-1]*(times[loc]-time)/gap
    else:
      gap=times[loc+1]-times[loc]
      speed=speeds[loc+1]*(time-times[loc])/gap+speeds[loc]*(times[loc+1]-time)/gap
    pointer+=speed
    time+=1/sr
  return output

def stutter(sample,start,reps,duration,sr=44100):
  start=int(start*sr)
  for i in range(reps):
    for j in range(int(duration*sr)):
      sample[int(i*duration*sr+j)+start]=sample[j+start]
  return sample

def cutToBeats(sample,bpm,beats,sr=44100):
  return sample[:int(beats*sr/(bpm/60))]
  
def lpf_stacked(a, sample, prev_states):
    for i in range(len(prev_states)):
        sample=prev_states[i]+a*(sample-prev_states[i])
        prev_states[i]=sample
    return sample

def filterWithAutomation(data,lowcut,hicut,times,sr=44100,order=5):
  data=np.asarray(data)
  n=len(data)
  t=np.arange(n)/sr
  cutoff=np.interp(t,times,hicut)
  out=[data[0]]
  for i in range(1,n):
    a=(2*math.pi*cutoff[i])/(2*math.pi*cutoff[i]+sr)
    out.append(lpf_stacked(a, data[i], out[i-min(i,4):]))
  return out

class synth:
  def __init__(self, sr, wavetype, a,d,s,r, bpm):
    self.sr=sr
    self.wavetype=wavetype
    self.a=a
    self.d=d
    self.s=s
    self.r=r
    self.bpm=bpm


  def noteToHz(self,note):
    note_map = {
        'C': 0, 'C#': 1, 'Db': 1,
        'D': 2, 'D#': 3, 'Eb': 3,
        'E': 4,
        'F': 5, 'F#': 6, 'Gb': 6,
        'G': 7, 'G#': 8, 'Ab': 8,
        'A': 9, 'A#': 10, 'Bb': 10,
        'B': 11
    }

    if len(note) == 2:
      pitch = note[0]
      octave = int(note[1])
    else:
      pitch = note[:2]
      octave = int(note[2])

    midi = (octave + 1) * 12 + note_map[pitch]

    n = midi - 69

    return 440 * (2 ** (n / 12))

  def noteToS(self,beats):
    bt=60/self.bpm
    return bt*beats

  def adsr(self,t):
    out=[]
    for i in range(round(t*self.sr)):
      if i<self.a*self.sr:
        out.append(i/(self.a*self.sr))
      elif i<self.a*self.sr+self.d*self.sr:
        j=i-self.a*self.sr
        out.append(self.s+(j-self.d*self.sr)*(self.s-1)/(self.d*self.sr))
      else:
        out.append(self.s)
    for i in range(0,min(int(self.r*self.sr),int(t*self.sr))):
      out[-1-i]*=i/int(self.r*self.sr)
    return out

  def oscillator(self, pitch, t, pw=0.5):
    t = np.arange(0,t,1/self.sr)

    if self.wavetype=="sine":
      return np.sin(2*np.pi*pitch*t)
    if self.wavetype=="saw":
      modded=t%(1/pitch)
      modded*=2*pitch
      return modded-1
    if self.wavetype=="square":
      modded=t%(1/pitch)
      modded=modded>pw/pitch
      return modded*2-1

  def synthesizeNote(self,note,beats,volume=1,pw=0.5):
    envellope=np.array(self.adsr(self.noteToS(beats)))
    oscillator=self.oscillator(self.noteToHz(note),self.noteToS(beats))
    while len(envellope)<len(oscillator):
      envellope=np.append(envellope,envellope[-1])
    while len(envellope)>len(oscillator):
      oscillator=np.append(oscillator,oscillator[-1])
    return oscillator*envellope*volume

class vocoder:
  def __init__(self,carrier,modulator,samplerate,bands,fmx,fmn,smear,order):
    carrier=list(carrier)
    while len(modulator)>len(carrier):
      carrier.append(0)
    self.carrier=carrier
    modulator=list(modulator)
    while len(modulator)<len(carrier):
      modulator.append(0)
    self.modulator=modulator
    self.sr=samplerate
    self.freqRange=(fmx,fmn)
    self.nbBands=bands
    r=(fmx/fmn)**(1/bands)
    edges=[fmn*(r**i) for i in range(bands+1)]
    self.bands=[[edges[i],edges[i+1]] for i in range(bands)]
    self.envellopes=[0]*bands
    self.smear=smear
    self.order=order

  def moving_average(self,x):
    window=int(self.smear*self.sr)
    x = np.asarray(x)

    padded = np.pad(x, (window - 1, 0), mode='constant', constant_values=0)

    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode='valid')

  def findEnvellopes(self):
    for i in range(self.nbBands):
      signal=butter_bandpass_filter(self.modulator,self.bands[i][1],self.bands[i][0],self.sr,order=self.order)
      signal=np.array(signal)
      signal=abs(signal)
      self.envellopes[i]=self.moving_average(signal)

  def applyEnvellopes(self):
    ret=np.zeros(len(self.carrier))
    for i in range(self.nbBands):
      signal=butter_bandpass_filter(self.carrier,self.bands[i][1],self.bands[i][0],self.sr,order=self.order)
      signal*=self.envellopes[i]
      ret+=signal
    return ret

  def vocode(self):
    self.findEnvellopes()
    return self.applyEnvellopes()

def pad(arr,d):
    base=[0]*d
    base.extend(arr)
    base.extend([0]*d)
    return base

def unpad(arr):
    i = 0
    while i < len(arr) and arr[i]==0:
        i+=1
    k = len(arr)-1
    while k >= 0 and arr[k]==0:
        k-=1
    return arr[i:k+1]

def adt(arr, d):
    newArr = pad(arr, d)
    transformed = [newArr[i] + newArr[i - d] for i in range(d, len(newArr))]
    return unpad(transformed)

def rev_adt(arr, d):
    padded_output = pad(arr, d)
    recovered = [0]*len(padded_output)
    for i in range(d, len(padded_output)):
        recovered[i] = padded_output[i] - recovered[i - d]
    return unpad(recovered)

def squareWave(freq, sampleRate, leng):
    ret = []
    amplitude = 2000
    samples = int(leng * sampleRate)
    period_samples = sampleRate / freq
    half_period = period_samples / 2

    curr = amplitude
    time_since_toggle = 0

    for i in range(samples):
        ret.append(curr)
        time_since_toggle += 1
        if time_since_toggle >= half_period:
            curr = -curr
            time_since_toggle = 0

    return ret
def sawWave(freq, sampleRate, leng):
  ret=[]
  curr=(sampleRate//freq)//2
  for i in range(int(sampleRate*leng)):
    curr-=1
    if curr==-(sampleRate//freq)//2:
      curr=(sampleRate//freq)//2
    ret.append(curr)
  return ret

def sineWave(freq, sampleRate, leng, amplitude=1000):
    ret = []
    for i in range(int(sampleRate * leng)):
        phase = 2 * math.pi * freq * (i / sampleRate)
        ret.append(math.sin(phase) * amplitude)
    return ret

def taper(arr,start,end):
    arr=np.asarray(arr,float)
    n=len(arr)
    start=min(int(start),n)
    end=min(int(end),n)
    if start:
        arr[:start]*=np.linspace(0,1,start,endpoint=False)
    if end:
        arr[-end:]*=np.linspace(1,0,end,endpoint=False)
    return arr

def custom_synth(freq, sampleRate, leng, harmonics):
    t = np.linspace(0, leng, int(sampleRate * leng), endpoint=False)
    wave = np.zeros_like(t)

    for multiplier, amplitude in harmonics:
        wave += amplitude * np.sin(2 * np.pi * freq * multiplier * t)

    return taper(wave, 0, sampleRate // 10)

def piano(freq, sampleRate, leng):
    t = np.linspace(0, leng, int(sampleRate * leng), endpoint=False)
    wave = 0.6 * np.sin(2 * np.pi * freq * t) + \
           0.3 * np.sin(2 * np.pi * freq * 2 * t) + \
           0.1 * np.sin(2 * np.pi * freq * 3 * t)

    return taper(wave,fs*0.005,fs*leng)
def toSemiTones(note):
    if len(note) == 2:
        note_part = note[0]
        octave = int(note[1])
    else:
        note_part = note[:2]
        octave = int(note[2])

    chromat = {
        "C": 0,
        "C#": 1, "Db": 1,
        "D": 2,
        "D#": 3, "Eb": 3,
        "E": 4,
        "F": 5,
        "F#": 6, "Gb": 6,
        "G": 7,
        "G#": 8, "Ab": 8,
        "A": 9,
        "A#": 10, "Bb": 10,
        "B": 11
    }

    semitones_from_C4 = (octave - 4) * 12 + chromat[note_part]
    semitones_from_A4 = semitones_from_C4 - 9
    return semitones_from_A4

def noteToFreq(note):
    semi = toSemiTones(note)
    return 440 * 2 ** (semi / 12)

def harmonica(freq, sampleRate, leng):
    t = np.linspace(0, leng, int(sampleRate * leng), endpoint=False)
    wave = np.zeros_like(t)

    for i in range(1, 30, 2):
        wave += (1.0 / i**0.8) * np.sin(2 * np.pi * freq * i * t)

    noise = np.random.normal(0, 1, len(t))

    filtered_noise = butter_bandpass_filter(noise, 1000, 3000, sampleRate)
    wave += 0.05 * filtered_noise

    tremolo = 1 + 0.2 * np.sin(2 * np.pi * 5 * t)
    wave *= tremolo

    attack = np.clip(t * 15, 0, 1)
    decay = np.exp(-t * 1.5)
    envelope = attack * decay
    wave *= envelope

    return taper(wave, 0, sampleRate // 50)




def violin(freq, sampleRate, leng):
    t = np.linspace(0, leng, int(sampleRate * leng), endpoint=False)
    wave = np.zeros_like(t)

    for i in range(1, 12):
        envelope = np.clip(t * 4, 0, 1)
        amp = (1.0 / i) * envelope
        wave += amp * np.sin(2 * np.pi * freq * i * t)

    env = np.clip(t * 3, 0, 1) * np.exp(-t * 1.5)
    wave *= env

    ret = wave
    return ret[::-1]


def flute(freq, sampleRate, leng):
    fluteHarmonics = [
        (1, 1.0),
        (2, 0.1),
        (3, 0.05)
    ]
    return taper(custom_synth(freq, sampleRate, leng, fluteHarmonics),fs//10,0)

class PedalSystem:
  def __init__(self, sampleRate=44100, rllvrStp=3, maxLen=4):
    """
    We define sample rate, an integer or float but hopefully an integer.
    We then define the start time as 0.
    We initialise the previous values array as empty.
    We set the base rollover steepness for when signal clipps
    """
    self.sampleRate = sampleRate
    self.time = 0
    self.prev = [0]*sampleRate*maxLen
    self.rllvrStp = rllvrStp
    self.useLater=[]
    self.mxTime = sampleRate*maxLen
    self.randTime = 0

  def overflow(self, n, maxVal, rolloverSteepness):
    """
    N is input value.
    Maxval is the maximum value that doesn't clip
    RolloverSteepness indicates how hard the clipping is if the signal clipps.
    Atan used so that the signal is clipped but not hard.
    """
    if rolloverSteepness==-1:
      rolloverSteepness=self.rllvrStp
    if n<maxVal and n>maxVal*-1:
      return n
    elif n>0:
      return maxVal+math.atan((n-maxVal)*rolloverSteepness/maxVal)/rolloverSteepness
    else:
      return -maxVal+math.atan((n+maxVal)*rolloverSteepness/maxVal)/rolloverSteepness

  def count(self, samples):
    """
    Increments time.
    Separated from the rest so two pedals don't increment time, making time move faster.
    Keeps a history of samples for use in other pedals.
    Should be used as the last pedal of a chain of pedals.

    Parameters:
    - samples: iterable of sample values
    - verb: if True, spreads the signal across multiple samples using Pascal's triangle
    - spread: the row of Pascal's triangle to use (number of samples = spread + 1)
    """

    for i in samples:
      self.prev[self.time % len(self.prev)] = i

      self.time += 1


  def tremolo(self, sample, depth, speed):
    """
    Applies tremolo to sample.
    Depth is how much the volume varies.
    Speed is how many times per second it fluctuates.
    """
    output=[]
    for k in range(len(sample)):
      lfo = math.sin(2 * math.pi * speed * (self.time+k) / self.sampleRate)
      y = sample[k] * (1 + depth * lfo)
      output.append(y)

    return output

  def overdrive(self, sample, multiplier, maxVal=1, rolloverSteepness=-1):
    """
    Applies overdrive to the samples.
    The multiplier is like the gain.
    MaxVal is when the signal starts to clip.
    The steepness is how hard the clip is. 0 is no clip (breaks due to /0) and -1 is set as default to resort to the default used in the setup.
    """
    output=[]
    for i in sample:
      output.append(self.overflow(i*multiplier,maxVal,rolloverSteepness))
    return output

  def vibrato(self, sample, depth=0.005, speed=5):
    """
    depth: vibrato amplitude in seconds (e.g., 0.005 = 5 ms)
    speed: LFO frequency in Hz
    """
    output = []
    max_delay = depth * self.sampleRate

    for k in range(len(sample)):
      t = (self.time + k) / self.sampleRate
      delay = max_delay * math.sin(2 * math.pi * speed * t)
      idx = (self.time + k - delay) % len(self.prev) - 1000
      idx_floor = int(math.floor(idx))
      idx_ceil = (idx_floor + 1) % len(self.prev)
      frac = idx - idx_floor
      y = (1 - frac) * self.prev[idx_floor] + frac * self.prev[idx_ceil]
      output.append(y)
    return output

  def delay(self, sample, time, divisionFactor=2):
    """
    The previous sound wave, time seconds earlier, is added to the original, divided by divisionFactor.
    Time has to be smaller than the mxLen. It must be larger than the sample length in seconds.
    """
    output=[]
    for i in range(len(sample)):
      t=self.time+i
      idx = int((t - time*self.sampleRate) % len(self.prev))
      out=sample[i] + self.prev[idx] / divisionFactor
      output.append(out)
    return output

  def noiseGate(self, sample, depth, maxVal=1):
    output=[]
    depth*=maxVal
    for i in sample:
      if i>depth or i<-depth:
        output.append(i)
      elif i>depth/2:
        output.append((i-depth/2)*2)
      elif i<-depth/2:
        output.append((i+depth/2)*2)
      else:
        output.append(0)
    return output

  def bitcrusher(self, sample, bits):
    """
    Cannot be 1
    """
    max_int = 2**(bits-1)-1

    output = []
    for y in sample:
      output.append(int(round(y * max_int)) / max_int)

    return output

def snare(pitch,pitchMix,noiseburstFreqMin,noiseburstFreqMax,top,fall,decay,attackSine,decaySine,time,sr=44100,order=5):
  t=np.arange(0,time,1/sr)
  pitch=np.sin(t*pitch*math.pi)
  for i in range(min(int(attackSine*sr),len(t))):
    pitch[i]*=i/min(attackSine*sr,len(t))
  for i in range(min(int(decaySine*sr),len(t))):
    pitch[-i-1]*=i/min(decaySine*sr,len(t))
  noise=[random.random()*2-1 for i in range(len(t))]
  for i in range(min(int(fall*sr), len(t))):
    env=1-(i/min(int(fall*sr), len(t)))
    noise[i] *= (top + (1 - top) * env)
  t = np.arange(len(noise)) / sr
  decay_env = np.exp(-5 * t / decay) if decay > 0 else np.ones(len(noise))
  noise *= decay_env
  noise=np.array(noise)
  return butter_bandpass_filter(noise,noiseburstFreqMin,noiseburstFreqMax,sr,order)+pitch*pitchMix

def hiHat(noiseburstFreqMin,noiseburstFreqMax,attack,decay,time,sr=44100,order=5):
  t=np.arange(0,time,1/sr)
  noise=[random.random()*2-1 for i in range(len(t))]
  noise=butter_bandpass_filter(noise,noiseburstFreqMin,noiseburstFreqMax,sr,order)
  for i in range(min(int(attack*sr),len(t))):
    noise[i]*=i/min(attack*sr,len(t))
  t = np.arange(len(noise)) / sr
  decay_env = np.exp(-5 * t / decay) if decay > 0 else np.ones(len(noise))
  noise *= decay_env
  noise=np.array(noise)
  return noise

def kick(freqStart,freqEnd,fmFreq,fmAmount,attack,decay,time,timeAfterSweep,grit=2,sr=44100,mult=6):
  t=np.arange(0,time,1/sr)
  freq=t
  freq = freqEnd * (t / time) + freqStart * (1 - t / time)
  t=np.arange(0,time+timeAfterSweep,1/sr)
  freq=np.concatenate([freq,np.zeros(int(timeAfterSweep*sr))+freqEnd])
  fm=np.array(waveFolder(np.sin(2*np.pi*fmFreq*t),1.3,mult,2,0.2))*fmAmount
  phase=2*np.pi*np.cumsum(freq+fm)/sr
  sound=np.sin(phase)
  for i in range(min(int(attack*sr),len(t))):
    sound[i]*=i/min(attack*sr,len(t))
  for i in range(min(int(decay*sr),len(t))):
    sound[-i-1]*=i/min(decay*sr,len(t))
  return waveFolder(sound,1,grit,2,0.2)

def reverbSingleBlock(minDelay,maxDelay,channels,sr=44100,scale=1):
  delays=[]
  for i in range(channels):
    delays.append(sympy.prime(random.randint(minDelay,maxDelay)))
  block=[0]*int(max(delays)*scale*sr)
  for i in delays:
    block[int(i*sr*scale)-1]=random.random()/channels
  return block

def reverbMultiblock(minDelay,maxDelay,channels,blocks,sr=44100,scale=1):
  rev=reverbSingleBlock(minDelay,maxDelay,channels,sr,scale)
  for i in range(blocks-1):
    rev=fftconvolve(rev,reverbSingleBlock(minDelay,maxDelay,channels,sr,scale),mode="full")
  return rev

def applyReverb(signal,minDelay,maxDelay,channels,blocks,mix,loCut,hiCut,sr=44100,scale=1):
  kernel=reverbMultiblock(minDelay,maxDelay,channels,blocks,sr,scale)
  signalNP=np.array(signal,dtype=float)
  for i in range(len(kernel)-1):
      signal.append(0)
  verbed=butter_bandpass_filter(np.array(fftconvolve(signal,kernel,mode="full")),loCut,hiCut,sr,order=4)
  verbed=verbed/max(verbed)
  dry=np.zeros(len(verbed))
  dry[:len(signalNP)]=signalNP
  dry=dry/max(dry)
  return verbed*mix+dry*(1-mix)
