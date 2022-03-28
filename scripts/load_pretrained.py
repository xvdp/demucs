"""
Models pretrained on /media/z/Malatesta/data/Audio/musdb18hq

Downloading: "https://dl.fbaipublicfiles.com/demucs/mdx_final/c511e2ab-fe698775.th" to /home/z/.cache/torch/hub/checkpoints/c511e2ab-fe698775.th
Downloading: "https://dl.fbaipublicfiles.com/demucs/mdx_final/7d865c68-3d5dd56b.th" to /home/z/.cache/torch/hub/checkpoints/7d865c68-3d5dd56b.th
Downloading: "https://dl.fbaipublicfiles.com/demucs/mdx_final/e51eebcc-c1b80bdd.th" to /home/z/.cache/torch/hub/checkpoints/e51eebcc-c1b80bdd.th
Downloading: "https://dl.fbaipublicfiles.com/demucs/mdx_final/a1d90b5c-ae9d2452.th" to /home/z/.cache/torch/hub/checkpoints/a1d90b5c-ae9d2452.th
Downloading: "https://dl.fbaipublicfiles.com/demucs/mdx_final/5d2d6c55-db83574e.th" to /home/z/.cache/torch/hub/checkpoints/5d2d6c55-db83574e.th
Downloading: "https://dl.fbaipublicfiles.com/demucs/mdx_final/cfa93e08-61801ae1.th" to /home/z/.cache/torch/hub/checkpoints/cfa93e08-61801ae1.th

$ demucs '/home/z/Music/U2 - Where The Streets Have No Name-3FsrPEUt2Dg.mp3'
Downloading: "https://dl.fbaipublicfiles.com/demucs/mdx_final/83fc094f-4a16d450.th" to /home/z/.cache/torch/hub/checkpoints/83fc094f-4a16d450.th
Downloading: "https://dl.fbaipublicfiles.com/demucs/mdx_final/464b36d7-e5a9386e.th" to /home/z/.cache/torch/hub/checkpoints/464b36d7-e5a9386e.th
Downloading: "https://dl.fbaipublicfiles.com/demucs/mdx_final/14fc6a69-a89dd0ee.th" to /home/z/.cache/torch/hub/checkpoints/14fc6a69-a89dd0ee.th
Downloading: "https://dl.fbaipublicfiles.com/demucs/mdx_final/7fd6ef75-a905dd85.th" to /home/z/.cache/torch/hub/checkpoints/7fd6ef75-a905dd85.th
Selected model is a bag of 4 models. You will see that many progress bars per track.


"""
from demucs import pretrained, separate

MODELS = ["0d19c1c6", "7ecf8ec1", "c511e2ab", "7d865c68"] # track a # 3 steps, not h
MODELS += ["e51eebcc", "a1d90b5c", "5d2d6c55", "cfa93e08"] # track b # 3 steps h
MODELS += ["83fc094f", "464b36d7", "14fc6a69", "7fd6ef75"] # ensemble for demucs built for system
#FINAL = '83fc094f-4a16d450'

TRACKS=["/home/z/Music/U2_Where_The_Streets_Have_No_Name.mp3", "/home/z/Music/Interstellar_Hans_Zimmer.mp3"]


def proc(track, model="83fc094f"):

    model = get_model(model)
    wav = separate.load_track(track, model.audio_channels, model.samplerate)

    ref = wav.mean(0)
    wav = (wav - ref.mean()) / ref.std()



def load_models():
    return {m:pretrained.get_model(m) for m in MODELS}


def print_model_properties(model=None):
    if model is None:
        for model in MODELS:
            print_model_properties(model)
    else:
        mod = get_model(model)
        
        print(f"{model}\t params: {get_nb_params(mod):,}\tchannels; {mod.audio_channels}\trate; {mod.samplerate}")

def get_model(model='83fc094f'):
    if isinstance(model, int):
        model = MODELS[model%len(MODELS)]
    return pretrained.get_model(model)

def get_nb_params(model):
    out = [p.numel() for p in model.parameters()]
    return sum(out)


"""
>>> print_model_properties()
0d19c1c6         params: 88,984,040     channels; 2     rate; 44100
7ecf8ec1         params: 88,984,040     channels; 2     rate; 44100
c511e2ab         params: 83,607,872     channels; 2     rate; 44100
7d865c68         params: 83,884,248     channels; 2     rate; 44100

e51eebcc         params: 83,637,832     channels; 2     rate; 44100
a1d90b5c         params: 83,633,984     channels; 2     rate; 44100
5d2d6c55         params: 83,633,984     channels; 2     rate; 44100
cfa93e08         params: 83,637,832     channels; 2     rate; 44100

83fc094f         params: 83,637,832     channels; 2     rate; 44100
464b36d7         params: 83,633,984     channels; 2     rate; 44100
14fc6a69         params: 83,633,984     channels; 2     rate; 44100
7fd6ef75         params: 83,637,832     channels; 2     rate; 44100


to run installr
demucs '/home/z/Music/Recording_Tests/Disc_1/05-Lola.flac'

demucs -n 0d19c1c6 '/home/z/Music/Recording_Tests/Disc_1/14-Lola (Mono Single Mix).flac'

"""