# Commits and commit messages

- Each commit should be a single logical change. Don't make several logical changes in one commit. For example, if a patch fixes a bug and optimizes the performance of a feature, split it into two separate commits.

- Each commit should be able to stand on its own, and each commit should build on the previous one. This way, if a commit introduces a bug, it should be easy to identify and revert.

- Each commit should be deployable and not break the build, tests, or functionality.

- If you ever amend, reorder, or rebase, your local branch will become divergent from the remote for the amended commit(s), so GitHub won't let you push. Simply force push to overwrite your old branch: `git push --force-with-lease`.

**We follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification for commit messages.**

- Each commit message should start with a type, followed by a scope (optional), and then a description. For example:
  - `feat(core): add new authentication module`
  - `fix(api): resolve issue with user login`
  - `docs(readme): update installation instructions`
- The entire commit message should be structured as follows:

  ```
  <type>(<scope>): <description>

  [optional body]

  [optional footer]
  ```

- The type should be one of the following:

  | Type     | Description                                                                        | Example                                                       |
  | -------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------- |
  | fix      | A bug fix in the project                                                           | `fix(docs): correct broken hyperlink in documentation`        |
  | feat     | New feature(s) being added to the project                                          | `feat(calendar): add new calendar feature`                    |
  | docs     | A new addition or update to the docs site and/or documentation                     | `docs(readme): add installation instructions`                 |
  | test     | New tests being added to the project                                               | `test(mobile): add Playwright tests for mobile docs site`     |
  | chore    | Tasks that don't modify any business logic code and that help maintain the project | `chore(deps): update project dependencies`                    |
  | style    | Changes that do not affect the meaning of the code and that improve readability    | `style(core): reformat code for readability`                  |
  | refactor | A change that improves the readability of code itself without modifying behavior   | `refactor(auth): extract helper function for validation`      |
  | perf     | A code change that improves code performance                                       | `perf(api): optimize data fetching logic`                     |
  | build    | Changes that affect the build system or external dependencies                      | `build(deps): upgrade to latest webpack version`              |
  | ci       | Changes to our CI configuration files and scripts                                  | `ci(github): add new workflow for automated testing`          |
  | revert   | Reverting a previous commit                                                        | `revert(core): revert commit abc123 that caused a regression` |

## Branching and pull requests

If there are multiple team members working on the project, it is best to create a new branch for each feature or bug fix. This way, the main branch remains stable and deployable.

The member working on the feature or bug fix can then create a pull request to merge their changes into the main branch. The pull request should be reviewed by at least one other team member before being merged.

Code should follow the existing style and conventions used in the project, not the style of the contributor. You shouldn't be able to identify who wrote a particular piece of code based on its style.
