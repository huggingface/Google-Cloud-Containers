.PHONY: docs clean help

docs: clean
	@echo "Processing README.md files from examples/gke, examples/cloud-run, and examples/vertex-ai..."
	@mkdir -p docs/source/examples
	@echo "Converting Jupyter Notebooks to MDX..."
	@doc-builder notebook-to-mdx examples/vertex-ai/notebooks/
	@echo "Auto-generating example files for documentation..."
	@python docs/scripts/auto-generate-examples.py
	@echo "Cleaning up generated Markdown Notebook files..."
	@find examples/vertex-ai/notebooks -name "vertex-notebook.md" -type f -delete
	@echo "Generating YAML tree structure and appending to _toctree.yml..."
	@echo "# GENERATED CONTENT DO NOT EDIT!" >> docs/source/_toctree.yml
	@echo "- sections:" >> docs/source/_toctree.yml
	@for dir in gke cloud-run vertex-ai; do \
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
		elif [ "$$dir" = "vertex-ai" ]; then \
			echo "      title: Vertex AI" >> docs/source/_toctree.yml; \
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
	@echo "Cleaning up generated Markdown Notebook files (if any)..."
	@find examples/vertex-ai/notebooks -name "vertex-notebook.md" -type f -delete
	@echo "Cleanup complete."

serve:
	@echo "Serving documentation via doc-builder"
	doc-builder preview gcloud docs/source --not_python_module

help:
	@echo "Usage:"
	@echo "  make docs   - Auto-generate the examples for the docs"
	@echo "  make clean  - Remove the auto-generated docs"
	@echo "  make help   - Display this help message"
