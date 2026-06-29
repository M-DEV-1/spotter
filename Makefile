SPARK      := zugzwang@spark-3100
SPARK_DIR  := ~/spotter
SPARK_VENV := /home/zugzwang/Desktop/mdev1-xplore/spotter/.venv/bin/activate
PORT       := 8888

.PHONY: setup push pull smoke episode serve clean

# one-time: symlink the menagerie + venv on the spark so paths resolve
setup:
	tailscale ssh $(SPARK) "\
		mkdir -p $(SPARK_DIR) && \
		ln -sfn ~/Desktop/mdev1-xplore/spotter/mujoco_menagerie $(SPARK_DIR)/mujoco_menagerie && \
		ln -sfn ~/Desktop/mdev1-xplore/spotter/.venv $(SPARK_DIR)/.venv"

# push code to spark (no menagerie, no outputs, no venv)
push:
	rsync -av --exclude='.git' --exclude='outputs' \
		--exclude='mujoco_menagerie' --exclude='.venv' \
		. $(SPARK):$(SPARK_DIR)/

# pull outputs back to local for viewing
pull:
	rsync -av $(SPARK):$(SPARK_DIR)/outputs/ ./outputs/
	@echo "episodes in outputs/episodes/"

# push + run smoke test on spark
smoke: push
	tailscale ssh $(SPARK) \
		"cd $(SPARK_DIR) && source $(SPARK_VENV) && MUJOCO_GL=egl python scripts/smoke_move.py"

# push + run one full pick-and-place episode on spark
episode: push
	tailscale ssh $(SPARK) \
		"cd $(SPARK_DIR) && source $(SPARK_VENV) && MUJOCO_GL=egl python scripts/run_episode.py"

# push + run rung 3 (perturbed unsupervised + supervised side-by-side)
rung3: push
	tailscale ssh $(SPARK) \
		"cd $(SPARK_DIR) && source $(SPARK_VENV) && MUJOCO_GL=egl python scripts/run_rung3.py"

# serve outputs over HTTP from spark — open http://spark-3100:PORT in browser
serve:
	@echo "open http://spark-3100:$(PORT)/"
	ssh -t $(SPARK) "cd $(SPARK_DIR)/outputs && python3 -m http.server $(PORT)"

clean:
	rm -rf outputs/smoke outputs/*.mp4
