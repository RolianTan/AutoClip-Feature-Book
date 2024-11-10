# AutoClip-Feature-Book

1. The demo inference result (automatically clipped video) can be found:
https://drive.google.com/file/d/1pKmR2JF7YHkeNnhvPVcEOipaU1sZ76jR/view?usp=drive_link

2. Here is the presentation slide:
https://docs.google.com/presentation/d/1D_ZEj6tLAETAT6sK1z0dqZLM6fKoUmzq/edit?usp=drive_link&ouid=112620541577719764728&rtpof=true&sd=true



The process of generate this video:
1. find a good mash-up video online (eg. https://drive.google.com/file/d/1BIAYRRFWV3lXB834nH627cZgd__5PNlN/view?usp=sharing)
2. use your own footage video (eg. https://drive.google.com/file/d/1VOrY-tx84wfpl9wIF_EF2sRa4StvRvIw/view?usp=sharing)
3. run the inference.py to make your own mash-up video with the similar editing style (eg. https://drive.google.com/file/d/1pKmR2JF7YHkeNnhvPVcEOipaU1sZ76jR/view?usp=drive_link)

# inference step
1. run feature_code_book_generator.py, this code will use GPT-Vision-API to help you find the best feature book for this mash-up. (you may need change the path where to load and save in the code)
2. run this command:
python inference.py \
    --exp_dir="YOUR TRAIN OUTPUT FOLDER" \
    --pretrained="best_model.pt" \
    --data_dir="WHERE THE FOOTAGE VIDEO IS" \
    --video_name="FOOTAGE VIDEO" \
    --json_name="YOUR JSON" \
    --fps=10 \
    --n_frames=8 \
    --size=224 \
    --out_mash_video="OUTPUT VIDEO NAME" \
    --reference_video="YOUR REFERENCE VIDEO" \
    --projection_dim=512
we provide a draft model weight (not fully trained, just for testing):
3. Enjoy the romance of GenAI!

# Training
1. prepare your training video in a video folder
2. run the video_caption_generator_nova.py, to use GPT-4o generate the training feature_code_book for training dataset (you will get a .JSON file)
3. then run the training code - main.py using:
 python main.py \
 --json_path="TRAINING DATA JSON"\
 --out_dir="TRAIN RESULT" \
 --n_frames=3 \
 --frame_size=224 \
 --batch_size=64 \
 --target_fps=5 \
 --epochs=50 \
 --lr=1e-5 \
 --projection_dim=512
   
# Analysis
Here are some of the output comparison during our testing 
At the same period of time in the video: 
Left is the auto-clipped mash-up video (in Japan)
Right is the oinline mash-up video (in Seatle, US)

1. Broad View Scenes
![Autoclipped Video](comparision/clip1.gif)
![Reference Video](comparision/clip1_ori.gif)

2. Buildings Scenes
![Autoclipped Video](comparision/clip2.gif)
![Reference Video](comparision/clip2_ori.gif)

3. People/Crowd Scenes
![Autoclipped Video](comparision/clip3.gif)
![Reference Video](comparision/clip3_ori.gif)

3. Dynamic Motion
![Autoclipped Video](comparision/clip4.gif)
![Reference Video](comparision/clip4_ori.gif)


