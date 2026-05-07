#Paste all of the source code from the main file in a cell in google colab and in the next cell paste this, and run all cells

#Create basic drum pattern, basic 4/4 backbeat
beat=np.concatenate([kick(200,40,40,0.3,0,0.1,0.1,0.4,grit=1.5,sr=44100,mult=6),snare(200,0.2,4000,7500,5,0.1,0.3,0.1,0.1,0.5,sr=44100,order=3)])
beat=np.concatenate([beat,kick(200,40,40,0.3,0,0.1,0.1,0.15,grit=1.5,sr=44100,mult=6),kick(200,40,40,0.3,0,0.1,0.1,0.15,grit=1.5,sr=44100,mult=6),snare(200,0.2,4000,7500,5,0.1,0.3,0.1,0.1,0.5,sr=44100,order=3)])

#Generate hi hats
hihats=list(hiHat(6000,9000,0,0.25,0.25,sr=44100,order=3))*8
beat=list(beat)

#Mix audio
for i in range(len(beat)):
  if i<len(hihats):
    beat[i]+=hihats[i]

#Make the beat 10 bars
beat*=10
#create bass synth
bass=synth(44100, "square", 0.1,0.1,0.3,0.05, 120)
#Sequence bass
bassLine=bass.synthesizeNote("G1",1,0.2,0.75)
bassLine=np.concatenate([bassLine,bass.synthesizeNote("D2",0.5,1,0.75),bass.synthesizeNote("B1",0.5,1,0.75)])
bassLine=np.concatenate([bassLine,list(bass.synthesizeNote("G1",0.5,1,0.75))*2,bass.synthesizeNote("D2",0.5,1,0.75),bass.synthesizeNote("B1",0.5,1,0.75)])
#Filter bass with 6dB/Oct for retaining some higher frequencies
bassLine=butter_bandpass_filter(bassLine,1,400,44100,1)
bassLine=list(bassLine)*10

#Make chords
pad=synth(44100, "saw", 0.2,0.2,0.3,0.05, 120)
chords=pad.synthesizeNote("G3",4,1)
chords+=pad.synthesizeNote("D3",4,1)
chords+=pad.synthesizeNote("B3",4,1)
chords=butter_bandpass_filter(chords,30,2000,44100)
chords=list(chords)*10

#Mix bassline, chords and drums and mute first bar of bass
for i in range(len(bassLine)//10):
  bassLine[i]=0
for i in range(len(beat)):
  if i<len(bassLine):
    beat[i]+=bassLine[i]
  if i<len(chords):
    beat[i]+=chords[i]

#Add stutter and varispeed and filter sweep
beat=stutter(beat,0.5*8,3,0.25,sr=44100)
beat=rhythmicVarispeed(beat,[1,0,0,-1,-1],[15,15.5,18,20,23])
beat=filterWithAutomation(beat,[0,10000],[0,5])

#Now you've read through the code, mess up the parameters and do whatever you want, have fun
Audio(beat,rate=44100)
