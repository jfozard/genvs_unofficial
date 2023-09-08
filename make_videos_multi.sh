p=output_view_sd_multi2
pp=Acars
checkpoint=output_sd_multi/large-k-multi-latest.pt
cfg=2
churn=1

python sample_views_multi.py --transfer=${checkpoint} --output_path ${p} --prefix ${pp} --cfg ${cfg} --churn ${churn} --n 4 --offset 2 --n_poses 40
python sample_views_multi.py --transfer=${checkpoint}  --output_path ${p} --prefix ${pp}_det --cfg ${cfg} --n 4 --offset 2 --n_poses 40 
python sample_views_multi.py --transfer=${checkpoint}  --output_path ${p} --prefix ${pp}2 --cfg ${cfg}  --churn ${churn} --n 4 --offset 13 --n_poses 40 
python sample_views_multi.py --transfer=${checkpoint}  --output_path ${p} --prefix ${pp}2_det --cfg ${cfg}  --n 4 --offset 13 --n_poses 40 

#exit 0

for i in {0..3}; do ffmpeg -y -r 10 -i ${p}/${pp}-sample-1-00000${i}-%d.png ${p}/${pp}-${i}.mp4; done
for i in {0..3}; do ffmpeg -y -r 10 -i ${p}/${pp}_det-sample-1-00000${i}-%d.png ${p}/${pp}_det-${i}.mp4; done
for i in {0..3}; do ffmpeg -y -i ${p}/${pp}-step-1-00000${i}-%d.png ${p}/${pp}-sample-${i}.mp4; done
for i in {0..3}; do ffmpeg -y -i ${p}/${pp}_det-step-1-00000${i}-%d.png ${p}/${pp}_det-sample-${i}.mp4; done
for i in {0..3}; do ffmpeg -y -r 10 -i ${p}/${pp}2-sample-1-00000${i}-%d.png ${p}/${pp}2-${i}.mp4; done
for i in {0..3}; do ffmpeg -y -r 10 -i ${p}/${pp}2_det-sample-1-00000${i}-%d.png ${p}/${pp}2_det-${i}.mp4; done
for i in {0..3}; do ffmpeg -y -i ${p}/${pp}2-step-1-00000${i}-%d.png ${p}/${pp}2-sample-${i}.mp4; done
for i in {0..3}; do ffmpeg -y -i ${p}/${pp}2_det-step-1-00000${i}-%d.png ${p}/${pp}2_det-sample-${i}.mp4; done



for cfg in 2.0; do
    for churn in 1.0; do 
	python sample_views_sd.py --transfer=${checkpoint}  --progress --output_path ${p} --prefix 100_uc_${cfg}_${churn} --unconditional --cfg ${cfg} --churn ${churn} --n 1 --n_poses 40 --steps 100
	ffmpeg  -y -r 10 -i ${p}/100_uc_${cfg}_${churn}-step-1-000000-%d.png ${p}/100_uc_${cfg}_${churn}.mp4
    done
done


for cfg in 3; do
    for churn in 1; do
	python sample_ar_multi.py --transfer=${checkpoint} --output_path ${p} --prefix cars_${cfg}_${churn} --steps 50 --n_poses 100 --cfg ${cfg} --churn ${churn} --n 4 --offset 13
	for i in {0..3}; do ffmpeg -y -i ${p}/cars_${cfg}_${churn}-sample-ar-1-00000${i}-%d.png ${p}/ar_${cfg}_${churn}-${i}.mp4; done
	python sample_ar_multi.py --transfer=${checkpoint} --output_path ${p} --prefix cars_b_${cfg}_${churn} --steps 50 --n_poses 100 --cfg ${cfg} --churn ${churn} --n 4
	for i in {0..3}; do ffmpeg -y -i ${p}/cars_b_${cfg}_${churn}-sample-ar-1-00000${i}-%d.png ${p}/ar_b_${cfg}_${churn}-${i}.mp4; done
    done
done
