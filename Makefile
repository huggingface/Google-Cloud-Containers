.PHONY: docs clean help

docs: clean
	@echo "Processing README.md files from examples/gke and examples/cloud-run..."
	@mkdir -p docs/source/examples
	@for dir in gke cloud-run; do \
		find examples/$$dir -name README.md | while read file; do \
			subdir=$$(dirname "$$file" | sed "s|examples/$$dir/||"); \
			base=$$(basename "$$subdir"); \
			if [ "$$file" = "examples/$$dir/README.md" ]; then \
				target="docs/source/examples/$$dir-index.mdx"; \
			else \
				target="docs/source/examples/$$dir-$$base.mdx"; \
			fi; \
			echo "Processing $$file to $$target"; \
			sed -E ' \
				s|\(\.\/imgs\/([^)]*\.png)\)|(https://raw.githubusercontent.com/huggingface/Google-Cloud-Containers/main/'"$${file%/*}"'/imgs/\1)|g; \
				s|\(\.\./([^)]+)\)|(https://github.com/huggingface/Google-Cloud-Containers/tree/main/examples/'"$$dir"'/\1)|g; \
				s|\(\.\/([^)]+)\)|(https://github.com/huggingface/Google-Cloud-Containers/tree/main/'"$${file%/*}"'/\1)|g; \
			' "$$file" > "$$target"; \
			sed -n -f docs/sed/huggingface-tip.sed "$$target" > "$$target.tmp" && mv "$$target.tmp" "$$target"; \
			sed -E 's/^(>[ ]*)+//g' "$$target" > "$$target.tmp" && mv "$$target.tmp" "$$target"; \
			if grep -qE '\(\.\./|\(\./' "$$target"; then \
				echo "WARNING: Relative paths still exist in the processed file."; \
				echo "The following lines contain relative paths, consider replacing those with GitHub URLs instead:"; \
				grep -nE '\(\.\./|\(\./' "$$target"; \
			else \
				echo "No relative paths found in the processed file."; \
			fi; \
		done; \
	done
	@echo "Generating YAML tree structure and appending to _toctree.yml..."
	@echo "# GENERATED CONTENT DO NOT EDIT!" >> docs/source/_toctree.yml
	@echo "- sections:" >> docs/source/_toctree.yml
	@for dir in gke cloud-run; do \
		echo "    - sections:" >> docs/source/_toctree.yml; \
		find docs/source/examples -name "$$dir-*.mdx" ! -name "$$dir-index.mdx" | sort | while read file; do \
			base=$$(basename "$$file" .mdx); \
			title=$$(head -n1 "$$file" | sed 's/^# *//'); \
			echo "        - local: examples/$$base" >> docs/source/_toctree.yml; \
			echo "          title: \"$$title\"" >> docs/source/_toctree.yml; \
		done; \
		echo "      isExpanded: false" >> docs/source/_toctree.yml; \
		echo "      local: examples/$$dir-index" >> docs/source/_toctree.yml; \
		if [ "$$dir" = "cloud-run" ]; then \
			echo "      title: Cloud Run" >> docs/source/_toctree.yml; \
		else \
			echo "      title: $$(echo $$dir | tr '[:lower:]' '[:upper:]')" >> docs/source/_toctree.yml; \
		fi; \
	done
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

serve:
	@echo "Serving documentation via doc-builder"
	doc-builder preview gcloud docs/source --not_python_module

help:
	@echo "Usage:"
	@echo "  make docs   - Auto-generate the examples for the docs"
	@echo "  make clean  - Remove the auto-generated docs"
	@echo "  make help   - Display this help message"
