# Completed experiments must remain visible

The read-visibility comparison finished all three stages, but its dashboard
stopped refreshing afterward. `checked_report()` rejected a reused adapter
because the session-local symlink resolved outside the session directory.
The adapter's exact target path and SHA-256 were already recorded in the frozen
configuration. This was a dashboard evidence-validation bug, not a new worker
crash or an interrupted training run.

The bridge now accepts such an external input only when the configuration still
matches its saved hash, the exact resolved file and digest are explicitly bound,
and the file's current contents match that digest. Changed weights, retargeted
links, unbound files and changed configurations remain rejected. Session paths
are canonicalized so a separate dashboard checkout can use existing session
storage without changing the experiment's frozen sources.

The regression tests reproduced both failures before the fix. All 39 desktop
bridge and finite-worker tests passed on the target Linux host with no skips.
The repaired bridge also reads the original completed comparison and its history
successfully, while the original experiment still passes its input and result
hash checks. The repair does not alter weights, grades, model prompts or limits.

The live Windows check exposed a second issue: the custom tab template omitted
the named selected-content host used by Windows accessibility. Status polling
worked, but page content was absent from the accessibility tree. The template
now supplies that standard part. A new UI regression check failed on the prior
template because the Home activity could not be found. Functional repair totals
also receive their own label instead of being presented as tested REPL totals.
The selected content lookup is defined in the upstream
[WPF tab accessibility implementation](https://github.com/dotnet/wpf/blob/main/src/Microsoft.DotNet.Wpf/src/PresentationFramework/System/Windows/Automation/Peers/TabItemAutomationPeer.cs).

The corrected Windows build passed the accessibility check, state controls,
background refresh, compact layout, Light/Dark/System contrast and every settings
page. The installed dashboard displays the completed rejected comparison, with
two observed refreshes. Version 0.1.4-preview.7 preserves the Dark appearance,
logo, worker processes, worker profiles and scheduled worker tasks.

An additional read-only inspection of the 2,180 training rows found 1,090 action
rows and 1,090 replay rows. Action targets contain 252 `write` calls, with no
`NotImplementedError` stub writes; 189 `finish` calls; 63 delegations; and 63
parent patch applications. Failed proposals are excluded from supervised targets,
and the trainer masks the conversation history. The candidate's observed stub
writes therefore do not establish that those failed proposals were supervised.
The curriculum uses scripted reference actions rather than corrections sampled
from this student's own failures. That is a limitation, not a demonstrated causal
explanation for its regression.

The [learning result](agent-read.md) remains rejected: functional solves 7/16 to
1/16, ordinary tested REPL 2/8 to 0/8, and genuine recursion 0/8 to 0/8. No new
training was run for this dashboard repair. The reserved heldouts remain unused,
and full Collect remains paused under the existing learning gates.
