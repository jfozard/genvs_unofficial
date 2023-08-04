p=output_view
python sample_views.py --transfer=../genvs-unofficial/genvs.pt  --progress --prefix cars --stochastic --n 2
python sample_views.py --transfer=../genvs-unofficial/genvs.pt  --progress --prefix cars_det --n 2
python sample_views.py --transfer=../genvs-unofficial/genvs.pt  --progress --prefix uc --unconditional --stochastic --n 1
python sample_views.py --transfer=../genvs-unofficial/genvs.pt  --progress --prefix uc_det --unconditional --n 1
#
for i in {0..1}; do ffmpeg -r 10 -i ${p}/cars-sample-1-00000${i}-%d.png ${p}/cars-${i}.mp4; done
for i in {0..1}; do ffmpeg -r 10 -i ${p}/cars_det-sample-1-00000${i}-%d.png ${p}/cars_det-${i}.mp4; done
for i in {0..1}; do ffmpeg -i ${p}/cars-step-1-00000${i}-%d.png ${p}/cars-sample-${i}.mp4; done
for i in {0..1}; do ffmpeg -i ${p}/cars_det-step-1-00000${i}-%d.png ${p}/cars_det-sample-${i}.mp4; done

ffmpeg  -i ${p}/uc-step-1-000000-%d.png ${p}/uc.mp4
ffmpeg  -i ${p}/uc_det-step-1-000000-%d.png ${p}/uc_det.mp4
python sample_ar.py --transfer=../genvs-unofficial/genvs.pt  --prefix cars 
for i in {0..9}; do ffmpeg -i ${p}/cars-sample-ar-1-00000${i}-%d.png ${p}/ar-${i}.mp4; done
