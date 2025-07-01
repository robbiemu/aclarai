In this PR, please audit the code:
- what is not completed?
- verify that there are no TODO or placeholders in the code/configuration
- check code quality -- good, easy to read pythonic code
- check comments and documentation added -- no references to anything in docs/project/ or the words "sprint 4" or sprint or epic. (these are changes in the PR, not PR descriptions or comments, etc). Verify also that there are no example or demo code.
- check for overreach -- no premature development that should be (better ) put off til later
- check test coverage - not in percentages but, are the functionality well tested? Do we have both mocked tests without the pytest integration decorator and tests against live services using the decorator (if this is appropriate for this feature)?
- consider the code in relation to the general development instructions in .github/copilot-instructions.md - do the research that this document suggests just as the developer should have. Does the code agree with the patterns and architecture we are embracing?