export function shouldReuseMountedDemo(
  canvasSelector: string,
  hasExistingMount: boolean,
  previousSource: string | undefined,
  nextSource: string,
  forceRemount: boolean,
): boolean {
  if (!hasExistingMount || forceRemount) return false;
  if (canvasSelector === "#demo-lines") return false;
  return previousSource === nextSource;
}
