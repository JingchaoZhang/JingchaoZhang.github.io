---
layout: single
author_profile: false
---


https://hcc-git.unl.edu/red-puppet.git
git clone git@hcc-git.unl.edu:red-puppet

Short summary
- hcc-git hosts various repos: red-puppet, puppet-sandhills-control, puppet-anvil-control, puppet-proj-control, etc.
- The repos are used by the puppet servers
- In a repo, a branch corresponds to a puppet environment
- The `production` branch is what is applied to clients by default
- You can run the puppet client against a different environment: `puppet agent --test --environment=blah`

Typical workflow:
- Create a new branch based on `production`
- Push changes
- Test: `puppet agent --test --environment=blah`
- Merge onto `production` and push
