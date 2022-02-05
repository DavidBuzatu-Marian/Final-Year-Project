import { getConfig } from '../config/defaultConfig';
import useSWR from 'swr';
import { fetcher } from './user';

export const useEnvironment = () => {
  const { data, mutate, error } = useSWR(
    getConfig()['environmentAddressesUrl'],
    fetcher
  );

  const loading = !data && !error;
  const environment = data && !error ? data : null;
  return [environment, { mutate, loading }, error];
};
