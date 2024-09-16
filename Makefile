.PHONY: docs clean help

docs:
	@echo "Processing README.md files from examples/gke..."
	@mkdir -p docs/source/examples
	@find examples/gke -name README.md | while read file; do \
		dir=$$(dirname "$$file"); \
		base=$$(basename "$$dir"); \
		if [ "$$file" = "examples/gke/README.md" ]; then \
			target="docs/source/examples/gke-index.mdx"; \
		else \
			target="docs/source/examples/gke-$$base.mdx"; \
		fi; \
		echo "Processing $$file to $$target"; \
		sed -E ' \
			s|\(\.\/imgs\/([^)]*\.png)\)|(https://raw.githubusercontent.com/huggingface/Google-Cloud-Containers/main/'"$$dir"'/imgs/\1)|g; \
			s|\(\.\./([^)]+)\)|(https://github.com/huggingface/Google-Cloud-Containers/tree/main/examples/gke/\1)|g; \
			s|\(\.\/([^)]+)\)|(https://github.com/huggingface/Google-Cloud-Containers/tree/main/'"$$dir"'/\1)|g; \
		' "$$file" > "$$target"; \
		if grep -qE '\(\.\./|\(\./' "$$target"; then \
			echo "WARNING: Relative paths still exist in the processed file."; \
			echo "The following lines contain relative paths, consider replacing those with GitHub URLs instead:"; \
			grep -nE '\(\.\./|\(\./' "$$target"; \
		else \
			echo "No relative paths found in the processed file."; \
		fi; \
	done
	@echo "Generating YAML tree structure and appending to _toctree.yml..."
	@echo "# GENERATED CONTENT DO NOT EDIT!" >> docs/source/_toctree.yml
	@echo "- sections:" >> docs/source/_toctree.yml
	@echo "    - sections:" >> docs/source/_toctree.yml
	@find docs/source/examples -name "gke-*.mdx" ! -name "gke-index.mdx" | sort | while read file; do \
		base=$$(basename "$$file" .mdx); \
		title=$$(head -n1 "$$file" | sed 's/^# *//'); \
		echo "        - local: examples/$$base" >> docs/source/_toctree.yml; \
		echo "          title: \"$$title\"" >> docs/source/_toctree.yml; \
	done
	@echo "      isExpanded: false" >> docs/source/_toctree.yml
	@echo "      local: examples/gke-index" >> docs/source/_toctree.yml
	@echo "      title: GKE" >> docs/source/_toctree.yml
	@echo "  # local: examples/index" >> docs/source/_toctree.yml
	@echo "  title: Examples" >> docs/source/_toctree.yml
	@echo "# END GENERATED CONTENT" >> docs/source/_toctree.yml
	@echo "YAML tree structure appended to docs/source/_toctree.yml"
	@echo "Documentation setup complete."

clean:
	@echo "Cleaning up generated documentation..."
	@rm -rf docs/source/examples
	@awk '/^# GENERATED CONTENT DO NOT EDIT!/,/^# END GENERATED CONTENT/{next} {print}' docs/source/_toctree.yml > docs/source/_toctree.yml.tmp && mv docs/source/_toctree.yml.tmp docs/source/_toctree.yml
	@echo "Cleanup complete."

help:
	@echo "Usage:"
	@echo "  make docs   - Auto-generate the examples for the docs"
	@echo "  make clean  - Remove the auto-generated docs"
	@echo "  make help   - Display this help message"
